from setuptools import setup, find_packages

setup(name='rmaddpg',
      version='0.0.1',
      description='Robust Multi-Agent Deep Deterministic Policy Gradient',
      url='https://github.com/SihongHo/Robust-MADDPG',
      author='Sihong He, Zhili Zhang, Songyang Han',
      author_email='sihong.he@uconn.edu',
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False,
      install_requires=['gym', 'numpy-stl']
)
