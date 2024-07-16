from setuptools import setup, find_packages

setup(
    name="sw_experiment",
    version="0.1",
    packages=find_packages(),
    entry_points={
        'console_scripts':['kuka_grasp_test=chaining_package.ENV.task_env.kuka_grasp_env:main','demon_data_gen=chaining_package.EXPERIMENT.demonstraton_data_gen.mimic_data_generator:main ']
    },
    include_package_data=True,
    install_requires=[
        # 列出所有依赖包
    ],
)