import os
from setuptools import setup

POPLAR_SDK_ENABLED = os.environ.get('POPLAR_SDK_ENABLED', default=None)
if POPLAR_SDK_ENABLED is None:
    raise Exception('Poplar SDK must be enabled in order to install poptorchlab')


def find_poptorch_wheel_path():
    POPLAR_SDK_ROOT = os.path.abspath(os.path.join(POPLAR_SDK_ENABLED, '..'))
    candidates = [*filter(lambda x: 'poptorch' in x, os.listdir(POPLAR_SDK_ROOT))]
    assert len(candidates) == 1, f'There must be exactly one file with "poptorch" in its name at {POPLAR_SDK_ROOT}, ' \
                                 f'but {len(candidates)} found. - {candidates}'
    return os.path.join(POPLAR_SDK_ROOT, candidates[0])


with open('requirements.txt', 'r') as f:
    version = '2.4.0'
    install_requires = f.readlines()
    poptorch_wheel_path = find_poptorch_wheel_path()
    print(f'Configured PopTorch path: {poptorch_wheel_path}')
    install_requires.append(
        f"poptorch @ file://localhost/{poptorch_wheel_path}#egg=poptorch-{version}"
    )

    setup(
        name='poptorchlab',
        version=version,
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
