from setuptools import setup
import os
import pip

POPLAR_SDK_ENABLED = os.environ.get('POPLAR_SDK_ENABLED', default=None)
if POPLAR_SDK_ENABLED is None:
    raise Exception('Poplar SDK must be enabled in order to install poptorchlab')

with open('requirements.txt', 'r') as f:
    install_requires = f.readlines()

    pip.main(['install', os.path.join(POPLAR_SDK_ENABLED, '..', 'poptorch*.whl')])

    setup(
        name='poptorchlab',
        version='2.4.0',
        author='Jiwoong Choi',
        author_email='jiwoongc@graphcore.ai',
        description='PopTorchLab - making experiments with PopTorch easier',
        long_description='file: README.md',
        long_description_content_type='text/markdown',
        url='https://github.com/jiwoong-choi/poptorchlab',
        project_urls={
            'Bug Tracker': 'https://github.com/jiwoong-choi/poptorchlab/issues'
        },
        packages=['poptorchlab'],
        zip_safe=False,
        install_requires=install_requires
    )
