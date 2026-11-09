@echo off
rem Starts the PWC-Net paper-replication queue runner (run_queue.py) with the cardioCad env.
rem Launch detached:  Start-Process -FilePath <this file> -WindowStyle Hidden
set "CONDA_PREFIX=C:\Users\andre\Anaconda3\envs\cardioCad"
set "CONDA_DEFAULT_ENV=cardioCad"
set "PATH=%CONDA_PREFIX%;%CONDA_PREFIX%\Library\mingw-w64\bin;%CONDA_PREFIX%\Library\usr\bin;%CONDA_PREFIX%\Library\bin;%CONDA_PREFIX%\Scripts;%PATH%"
set "MKL_THREADING_LAYER=TBB"
set "PYTHONUNBUFFERED=1"
cd /d D:\PHD\Development\Python\Cardio_CAD
"%CONDA_PREFIX%\python.exe" -m motion_estimator.methods.pwcnet.run_queue %* >> "D:\PHD\Development\Python\Cardio_CAD\motion_estimator\artifacts\paper-runs\runner.out" 2>&1
