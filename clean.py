
import io
import os
import time
from typing import Iterable
from typing import Optional
from typing import Tuple
import threading
import queue
import gc
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

import tarfile
import orjson
import logging
import re
import csv

from collections import defaultdict

import pandas as pd
import mmh3
import click
import boto3
import botocore.config

from PIL import Image


logging.basicConfig(
    format='%(asctime)s [%(levelname)s] [%(module)s:%(lineno)d] %(message)s',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


BATCH_SIZE = 10000
GB = 1024 * 1024 * 1024
THREAD_NUMBER = 128


class Storage(object):

    def __init__(self, access_key: str, secret_key: str, bucket_name: str) -> None:
        self._client = boto3.resource(
            's3',
            endpoint_url='http://aoss-internal.cn-sh-01.sensecoreapi-oss.cn',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=botocore.config.Config(max_pool_connections=THREAD_NUMBER * 2)
        )
        self._bucket = self._client.Bucket(bucket_name)

    def list(self, prefix: str) -> Iterable[str]:
        """返回当前前缀的所有文件名"""
        for obj in self._bucket.objects.filter(Prefix=prefix):
            yield obj.key

    def get(self, name: str) -> bytes:
        """获取制定文件的内容"""
        for i in range(3):
            try:
                with io.BytesIO() as data:
                    self._bucket.download_fileobj(name, data)
                    return data.getvalue()
            except Exception as e:
                logger.info('download %s retry %s', name, i)
                time.sleep(5)

    def put(self, name: str, data: bytes):
        """上传文件到指定目录"""
        for i in range(3):
            try:
                with io.BytesIO(data) as data:
                    data.seek(0)
                    self._bucket.upload_fileobj(data, Key=name)
                    return
            except Exception as e:
                logger.info('upload %s retry %s', name, i)
                time.sleep(5)


class Utils(object):

    @classmethod
    def covert_to_jpg(cls, data: bytes) -> bytes:
        """图片格式转换为jpg"""
        with io.BytesIO(data) as image_file:
            image_file.seek(0)

            im = Image.open(image_file)
            if im.format == 'JPEG':
                im.close()
                return image_file.getvalue()
        new_im = im.convert('RGB')
        im.close()

        with io.BytesIO() as jpg_file:
            new_im.save(jpg_file, format='JPEG', quality=95)
            new_im.close()
            return jpg_file.getvalue()

    @classmethod
    def compute_hash(cls, url: str, text: str) -> str:
        """计算数据对应的文件名称"""
        url = url or ''
        text = text or ''
        url_text = (url + text).encode('utf-8')
        return mmh3.hash64(url_text)[0]


class TarWriter(object):

    def __init__(self, name: str) -> None:
        self.name = name
        self._lock = threading.Lock()
        self._tar = tarfile.TarFile(name=name, mode='w')

    def close(self):
        self._tar.close()

    def remove(self):
        if os.path.exists(self.name):
            os.remove(self.name)

    def read(self) -> bytes:
        with open(self.name, 'rb') as f:
            return f.read()

    def add(self, name: str, data: dict, image: bytes):
        # json 文件
        json_value = orjson.dumps(data)
        json_file = io.BytesIO(json_value)
        json_file.seek(0)
        json_file_info = tarfile.TarInfo(name=name + '.json')
        json_file_info.size = len(json_value)

        # text 文件
        text_value = data['TEXT'].encode('utf-8')
        text_file = io.BytesIO(text_value)
        text_file.seek(0)
        text_file_info = tarfile.TarInfo(name=name + '.txt')
        text_file_info.size = len(text_value)

        # 图片文件
        image_value = image
        image_file = io.BytesIO(image_value)
        image_file.seek(0)
        image_file_info = tarfile.TarInfo(name=name + '.jpg')
        image_file_info.size = len(image_value)

        with self._lock:
            self._tar.addfile(tarinfo=json_file_info, fileobj=json_file)
            self._tar.addfile(tarinfo=text_file_info, fileobj=text_file)
            self._tar.addfile(tarinfo=image_file_info, fileobj=image_file)
        
        json_file.close()
        text_file.close()
        image_file.close()
        del json_file
        del text_file
        del image_file


class Processor(object):

    def __init__(
            self,
            source_storage: Storage,
            target_storage: Storage,
            prefix: str,
            metadata: str,
    ) -> None:
        """
        :source_storage
        :prefix:
            LAION5B/LAION-5B/laion2B-en/00000/
        :metadata
            LAION5B/metadata/laion2B-en/part-00000-5114fd87-297e-42b0-9d11-50f1df323dfa-c000.snappy.parquet
        """
        self._lock = threading.Lock()
        self._source = source_storage
        self._target = target_storage
        self._part_number = re.findall(r'/part-(\d+)-', metadata)[0]
        self._prefix = prefix
        self._metadata = metadata
        self._tar_writer: Optional[TarWriter] = None

        self._error_file = open(os.path.join('./error', metadata.replace(
            '/', '_') + '.error.jsonl'), 'wb')

    def close(self):
        self._error_file.close()

    def download_parquet(self) -> str:
        """下载metadata文件到本地"""
        logger.info('begin download parquet. %s', self._metadata)
        start = time.time()
        filename = os.path.join('./parquet', self._metadata.replace('/', '_'))
        if not os.path.exists(filename):
            parquet_data = self._source.get(self._metadata)
            with open(filename, 'wb') as f:
                f.write(parquet_data)
            logger.info('success download parquet. %s, size=%s GB, cost=%ss',
                        self._metadata, len(parquet_data) / GB, time.time() - start)
            del parquet_data
        return filename

    def parse_parquet(self, filename: str) -> Iterable[Tuple[int, dict]]:
        """解析parquet，返回map数据"""
        start = time.time()
        logger.info('begin parse parquest')
        df = pd.read_parquet(filename)
        logger.info('finish parse parquet, cost=%ss', time.time() - start)
        for row_index, row in df.iterrows():
            data = {
                'SAMPLE_ID': row['SAMPLE_ID'],
                'URL': row['URL'],
                'TEXT': row['TEXT'] or '',
                'HEIGHT': row['HEIGHT'],
                'WIDTH': row['WIDTH'],
                'LICENSE': row['LICENSE'],
                'NSFW': row['NSFW'],
                'similarity': row['similarity'],
            }
            yield row_index, data

    def get_image_filename(self, data: dict) -> str:
        """通过文件名，获取完成的文件路径"""
        fn = str(data['SAMPLE_ID'])
        try:
            for filename in self._source.list(prefix=os.path.join(self._prefix, fn)):
                return filename
        except Exception as e:
            pass
        fn = Utils.compute_hash(url=data['URL'], text=data['TEXT'])
        try:
            for filename in self._source.list(prefix=os.path.join(self._prefix, fn)):
                return filename
        except Exception as e:
            pass

    def upload_tar(self, index: int, row_index: int):
        if not self._tar_writer:
            return
        self._tar_writer.close()
        # 上传到 OSS
        target_path = os.path.join(
            'LAION5B-clean/',
            self._prefix.replace('LAION5B/LAION-5B/', ''),
            os.path.split(self._tar_writer.name)[1],
        )
        data = self._tar_writer.read()
        start = time.time()
        logger.info('begin upload tar to=%s, size=%s',
                    target_path, len(data) / GB)
        self._target.put(target_path, data)
        logger.info('finish upload tar to %s, cost=%s(s)',
                    target_path, time.time() - start)
        logger.info('last index=%s row_index=%s', index, row_index)
        self._tar_writer.remove()
        self._tar_writer = None
        gc.collect()

    def handle_error(self, data: dict):
        # 处理图片不存在情况, 写入 csv 文件
        data['_IMAGE_ID'] = Utils.compute_hash(url=data['URL'], text=data['TEXT'])

        with self._lock:
            self._error_file.write(orjson.dumps(data) + b'\n')
            self._error_file.flush()

    def handle_image(self, index: int, row_index: int, data: dict, image: bytes) -> int:
        batch, position = divmod(index, BATCH_SIZE)
        if index % 100 == 0:
            logger.info('%s-%08d-%05d %d', self._part_number, batch, position, row_index)

        if not self._tar_writer:
            self._tar_writer = TarWriter(name='./tar/%s-%08d.tar' % (self._part_number, batch),)

        self._tar_writer.add(
            name='%08d-%05d' % (batch, position),
            data=data,
            image=image,
        )

        if (position + 1) % BATCH_SIZE == 0:
            self.upload_tar(index, row_index)

        return index + 1

    def _handle_data(self, row_index: int, data: dict, data_queue: queue.Queue):
        image = None
        try:
            image_filename = self.get_image_filename(data)
            if image_filename:
                image = self._source.get(image_filename)
                image = Utils.covert_to_jpg(image)
        except Exception as e:
            pass
        data_queue.put((row_index, data, image))

    def _handle(self, start_index: int, data_queue: queue.Queue):
        index = start_index
        while True:
            job = data_queue.get()
            if not job:
                break
            data_queue.task_done()
            row_index, data, image = job
            if not image:
                self.handle_error(data=data)
                continue

            index = self.handle_image(index=index, row_index=row_index, data=data, image=image)

        # 处理最后一个批次
        self.upload_tar()


    def run(self, start_index: int = 0, last_row_index: int = -1):
        pool = ThreadPoolExecutor(max_workers=THREAD_NUMBER)
        pool._work_queue = queue.Queue(THREAD_NUMBER)
        data_queue = queue.Queue(THREAD_NUMBER)

        main_thread = threading.Thread(target=self._handle, args=(start_index, data_queue, ))
        main_thread.start()

        filename = self.download_parquet()
        for row_index, data in self.parse_parquet(filename=filename):
            if row_index < last_row_index:
                continue
            pool.submit(self._handle_data, row_index, data, data_queue)
        
        data_queue.put(None)
        main_thread.join()
        # 删除本地的 metadata 文件
        os.remove(filename)


sensetime = Storage(
    access_key='9266E6C1392249CD8F2C7CD223F5203E',
    secret_key='613A95404D9744FDBE12921FD0907962',
    bucket_name='aidmpsys',
)

opendata = Storage(
    access_key='9266E6C1392249CD8F2C7CD223F5203E',
    secret_key='613A95404D9744FDBE12921FD0907962',
    bucket_name='agishared1',
)


def init_logger(log_filename: str):
    logging.basicConfig(
        format='%(asctime)s [%(levelname)s] [%(module)s:%(lineno)d] %(message)s',
        level=logging.INFO,
        handlers=[
            logging.FileHandler('./log/%s.log' % log_filename),
            logging.StreamHandler()
        ],
        force=True,
    )

    logger.info('PID=%s', os.getpid())


@click.group(name='agi')
def agi():
    pass


@agi.command(name='list', help='获取 aidmpsys bucket 下的文件列表')
@click.option('--prefix', required=True)
def list_prefix(prefix: str):
    start = time.time()
    for name in sensetime.list(prefix=prefix):
        print(name)
    print(time.time() - start)


@agi.command(name='metadata', help='获取 agishared1 bucket 下的文件列表')
@click.option('--prefix', default='LAION5B-clean/', required=True)
def list_metadata(prefix: str):
    start = time.time()
    c = 0
    times = []
    for name in opendata.list(prefix=prefix):
        c += 1
        if c % 100:
            times.append(time.time() - start)
        print(name)
    print(times)
    print(time.time() - start)


@agi.command(name='dl', help='下载 agishared1 bucket 下的文件')
@click.option('--name', required=True)
@click.option('--target', default='.')
def download_metadata(name: str, target: str):
    start = time.time()
    _, filename = os.path.split(name)
    data = opendata.get(name=name)
    with open(os.path.join(target, filename), 'wb') as f:
        f.write(data)
    time_cost = time.time() - start
    mb = len(data) / 1024 / 1024
    speed = mb / time_cost
    print('Total: %s MB, Time Cost: %ss, Speed: %f MB/s' %
          (mb, time_cost, speed))


@agi.command(name='download', help='下载 aidmpsys bucket 下的文件到本地')
@click.option('--name', required=True)
@click.option('--target', default='.')
def download(name: str, target: str):
    start = time.time()
    _, filename = os.path.split(name)
    data = sensetime.get(name=name)
    with open(os.path.join(target, filename), 'wb') as f:
        f.write(data)
    time_cost = time.time() - start
    mb = len(data) / 1024 / 1024
    speed = mb / time_cost
    print('Total: %s MB, Time Cost: %ss, Speed: %f MB/s' %
          (mb, time_cost, speed))


@agi.command(name='run', help='运行数据清洗打包')
@click.option('--prefix', required=True, help="图片目录")
@click.option('--metadata', required=True, help="元数据")
@click.option('--row_index', default=-1, help='上一次数据执行到的row_index')
@click.option('--start_index', default=0, help='表示从那个 block 开始计算')
def run(prefix: str, metadata: str, row_index: int = 0, start_index: int = 0):
    init_logger(metadata.replace('/', '_'))

    p = Processor(
        source_storage=sensetime,
        target_storage=opendata,
        prefix=prefix,
        metadata=metadata,
    )
    p.run(start_index, row_index)


# @agi.command(name='split', help='将 parquet 文件按 10000 切分为不同的文件')
# @click.option('--prefix', required=True, help="图片目录")
# @click.option('--metadata', required=True, help="元数据")
# def split(prefix: str, metadata: str):
#     init_logger(prefix=prefix, metadata=metadata)
#     p = Processor(
#         source_storage=sensetime,
#         target_storage=opendata,
#         prefix=prefix,
#         metadata=metadata,
#     )
#     p.split()


def _worker(prefix: str, metadata: str):
    os.system(f'python3 agi.py run --prefix="{prefix}" --metadata="{metadata}"')


@agi.command(name='multi', help='多进程')
@click.option('--start', type=int, required=True, help='起始编号(包含)')
@click.option('--end', type=int, required=True, help='结束编号(不包含)')
@click.option('--processes', type=int, default=8, required=True, help='进程数')
def multi(start: int, end: int, processes: int = 8):

    logger.info('start')
    cache = {}
    for name in sensetime.list(prefix='LAION5B/metadata/laion2B-en/'):
        part_number = re.findall(r'/part-(\d+)-', name)[0]

        prefix = f'LAION5B/LAION-5B/laion2B-en/{part_number}/'
        metadata = name

        cache[int(part_number)] = {'prefix': prefix, 'metadata': metadata}
    
    results = []
    pool = multiprocessing.Pool(processes=processes)
    for index in range(start, end):
        logger.info('run %s', index)
        results.append(pool.apply_async(_worker, kwds=cache[index]))

    logger.info('wait')
    [r.get() for r in results]

    # 关闭进程池
    pool.close()
    # 等待所有任务完成
    pool.join()


@agi.command(name='gen', help='生成打包命令')
def gen():
    for name in sensetime.list(prefix='LAION5B/metadata/laion2B-en/'):
        part_number = re.findall(r'/part-(\d+)-', name)[0]

        prefix = f'LAION5B/LAION-5B/laion2B-en/{part_number}/'
        metadata = name

        print(f'python3 agi.py run --prefix="{prefix}" --metadata="{metadata}"')

if __name__ == '__main__':
    agi()

"""

### list prefix files

python3 agi.py list --prefix=LAION5B/LAION-5B/laion2B-en/00000/8242026769295241287

### run

python3 agi.py run --prefix="LAION5B/LAION-5B/laion2B-en/00000/" --metadata="LAION5B/metadata/laion2B-en/part-00000-5114fd87-297e-42b0-9d11-50f1df323dfa-c000.snappy.parquet"

"""



