@echo OFF
rem When launching Python scripts from Task Scheduler with a conda interpreter, it is necessary to activate the conda environment, which is best done using a batch script.
rem If the environment is not activated, Task Scheduler will show the task has executed correctly, but the code will not run.
rem Source: https://gist.github.com/maximlt/531419545b039fa33f8845e5bc92edd6

rem This solution does not require:
rem - conda to be in the PATH
rem - cmd.exe to be initialized with conda init

rem Define here the path to your conda installation
set CONDAPATH=C:\Users\russelbp\AppData\Local\anaconda3
rem Define here the name of the environment
set ENVNAME=dfm_tools_env

rem The following command activates the base environment.
if %ENVNAME%==base (set ENVPATH=%CONDAPATH%) else (set ENVPATH=%CONDAPATH%\envs\%ENVNAME%)

rem Activate the conda environment
rem Using call is required here, see: https://stackoverflow.com/questions/24678144/conda-environments-and-bat-files
call %CONDAPATH%\Scripts\activate.bat %ENVPATH%

rem Run a python script in that environment
python ScrapeStep.py

rem Deactivate the environment
call conda deactivate

rem Other approaches: if conda is directly available from the command line then the following code works.
rem call activate someenv
rem python script.py
rem conda deactivate

rem One could also use the conda run command
rem conda run -n someenv python script.py