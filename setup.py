from setuptools import setup

setup(
    name='poptorchlab',
    version='2.4.0',
    author='Jiwoong Choi',
    author_email='jiwoongc@graphcore.ai',
    description='PopTorchLab - making experiments with PopTorch easier',
    long_description='file: README.md',
    long_description_content_type='text/markdown',
    url='https://github.com/jiwoong-choi/poptorchlab',
    project_urls='Bug Tracker = https://github.com/jiwoong-choi/poptorchlab/issues',
    packages=['poptorchlab'],
    zip_safe=False,
    install_requires=['torch==1.10.0+cpu'],
    dependency_links=['https://download.pytorch.org/whl/torch_stable.html']
)
