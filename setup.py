from setuptools import setup, find_packages

setup(
    name="chaining_package",
    version="0.1",
    packages=find_packages(),
    entry_points={
        'console_scripts':['kuka_grasp_test=ENV.task_env.kuka_grasp_env']
    },
    include_package_data=True,
    install_requires=[
        # 列出所有依赖包
    ],
)