```python
import matplotlib.pyplot as plt
from IPython import display
import gymnasium as gym
```


```python
def render_env(env):
    plt.imshow(env.render())
    plt.axis('off')
    display.display(plt.gcf())
    display.clear_output(wait=True)
```


```python
env = gym.make("MountainCar-v0", render_mode="rgb_array")
env.reset()
render_env(env)
```

```mermaid
graph TD
    考试前 -->|100%| 游戏
    考试前 -->|100%| 复习

    游戏 -->|R:+10| 愉快
    愉快 -->|100%| 考试
    复习 -->|R:-20| 悲伤
    悲伤 -->|100%| 考试

    考试 -->|100%| 挂科
    考试 -->|10%| 及格
    考试 -->|80%| 及格并拿到零用钱
    考试 -->|10%| 及格并拿到零用钱

    挂科 -->|R:-5| 考试前
    及格 -->|R:+10| 考试前
    及格并拿到零用钱 -->|R:+100| 考试前


```python
!jupyter nbconvert --to markdown render_jupyter.ipynb

```

    Traceback (most recent call last):
      File "C:\ProgramData\Anaconda3\envs\makemerich\lib\runpy.py", line 196, in _run_module_as_main
        return _run_code(code, main_globals, None,
      File "C:\ProgramData\Anaconda3\envs\makemerich\lib\runpy.py", line 86, in _run_code
        exec(code, run_globals)
      File "C:\ProgramData\Anaconda3\envs\makemerich\Scripts\jupyter-nbconvert.EXE\__main__.py", line 4, in <module>
      File "C:\ProgramData\Anaconda3\envs\makemerich\lib\site-packages\nbconvert\nbconvertapp.py", line 193, in <module>
        class NbConvertApp(JupyterApp):
      File "C:\ProgramData\Anaconda3\envs\makemerich\lib\site-packages\nbconvert\nbconvertapp.py", line 252, in NbConvertApp
        Options include {get_export_names()}.
      File "C:\ProgramData\Anaconda3\envs\makemerich\lib\site-packages\nbconvert\exporters\base.py", line 145, in get_export_names
        e = get_exporter(exporter_name)(config=config)
      File "C:\ProgramData\Anaconda3\envs\makemerich\lib\site-packages\nbconvert\exporters\base.py", line 106, in get_exporter
        exporter = items[0].load()
      File "C:\ProgramData\Anaconda3\envs\makemerich\lib\importlib\metadata\__init__.py", line 171, in load
        module = import_module(match.group('module'))
      File "C:\ProgramData\Anaconda3\envs\makemerich\lib\importlib\__init__.py", line 126, in import_module
        return _bootstrap._gcd_import(name[level:], package, level)
      File "C:\ProgramData\Anaconda3\envs\makemerich\lib\site-packages\jupyter_contrib_nbextensions\__init__.py", line 5, in <module>
        import jupyter_nbextensions_configurator
      File "C:\ProgramData\Anaconda3\envs\makemerich\lib\site-packages\jupyter_nbextensions_configurator\__init__.py", line 18, in <module>
        from notebook.base.handlers import APIHandler, IPythonHandler
    ModuleNotFoundError: No module named 'notebook.base'
    


```python

```
