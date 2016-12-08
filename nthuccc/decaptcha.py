import argparse
import contextlib
import glob
import gzip
import os
import pickle
import time

from sklearn import svm, preprocessing, pipeline
from scipy.misc import imread
from nthuccc import NTHUCCCU
import numpy


@contextlib.contextmanager
def timethis(formatter):
    start = time.perf_counter()
    yield
    stop = time.perf_counter()
    print(formatter.format(duration=stop - start))


s0 = numpy.s_[2:15]
s1 = numpy.s_[15:28]
s2 = numpy.s_[28:41]


def bw_imread(name):
    im = imread(name, 'L')
    im[im <= 160] = 0
    im[im > 160] = 255
    return im


def extract_images(images):
    return [
        numpy.array(
            [image[..., sub] for image in images], dtype=float
        ).reshape((len(images), -1)) for sub in [s0, s1, s2]
    ]


def extract_target(target):
    return (
        numpy.array([t // 100 for t in target]),
        numpy.array([t // 10 % 10 for t in target]),
        numpy.array([t % 10 for t in target]),
    )


def read_images(filenames, threshold=160):
    images = []
    target = []
    for i, fullname in enumerate(filenames, start=1):
        dir_, _, filename = fullname.rpartition('/')
        answer, _, suffix = filename.partition('-')
        images.append(bw_imread(fullname))
        target.append(int(answer))
        print('Loading images... ({}/{})'.format(i, len(filenames)), end='\r')
    print()
    return images, target


def get_pipeline():
    return pipeline.Pipeline(
        [
            ('scaler', preprocessing.StandardScaler()),
            ('svc', svm.SVC()),
        ]
    )


def train(images3, target3):
    return tuple(
        get_pipeline().fit(image, target)
        for image, target in
        zip(extract_images(images3), extract_target(target3))
    )


_DEFAULT_URL = \
    'https://www.ccxp.nthu.edu.tw/ccxp/INQUIRE/JH/6/6.2/6.2.9/JH629001.php'


def decaptcha(pipelines, name):
    images = extract_images((bw_imread(name), ))
    r0, r1, r2 = [
        classifier.predict(image)
        for (classifier, image) in zip(pipelines, images)
    ]
    return '{}{}{}'.format(r0[0], r1[0], r2[0])


class DecaptchaFailure(NTHUCCCU):
    '''
    Failing to decaptcha
    '''


def get(pipelines, form_url=_DEFAULT_URL, _verify=True):
    from urllib.request import urlopen
    from urllib.parse import urljoin
    import lxml.html
    import requests
    # lxml.html.parse does not work somehow
    html = lxml.html.fromstring(requests.get(form_url).content)
    form, = html.xpath('//form')
    acixstore, = form.xpath('//input[@name="ACIXSTORE"]/@value')
    captcha = decaptcha(
        pipelines,
        urlopen(
            'https://www.ccxp.nthu.edu.tw/ccxp/INQUIRE/JH/mod/auth_img/'
            'auth_img.php?ACIXSTORE={}'.format(acixstore)
        )
    )
    if _verify:
        form_action_url = urljoin(
            form_url, form.attrib.get('action', form_url)
        )
        response = requests.post(
            form_action_url, {'ACIXSTORE': acixstore,
                              'auth_num': captcha}
        )
        if b'interrupted' in response.content:
            raise DecaptchaFailure(
                'Interrupted session for {}'.format(acixstore)
            )
        if b'Wrong check code' in response.content:
            raise DecaptchaFailure('Wrong check code for {}'.format(acixstore))
    return acixstore, captcha


def benchmark(pipelines, n):
    success = 0
    for i in range(n):
        try:
            acixstore, captcha = get(pipelines)
        except DecaptchaFailure as e:
            print(e)
        else:
            success += 1
            print(acixstore, captcha)
    return success


def load_pipelines(filename):
    with gzip.open(filename) as file:
        return pickle.load(file)


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('imgdir')
    parser.add_argument('output')
    parsed_args = parser.parse_args(args)
    filenames = glob.glob(os.path.join(parsed_args.imgdir, '*.png'))
    images, target = read_images(filenames)
    with timethis('Fitting took {duration:.4g} seconds'):
        classifiers = train(images, target)
    with gzip.open(parsed_args.output, mode='wb') as file:
        pickle.dump(classifiers, file)


if __name__ == '__main__':
    main()
