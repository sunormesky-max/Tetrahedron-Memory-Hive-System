@echo off
cd /d C:\龙虾\Tetrahedron-Memory-Hive-System
:loop
python C:\龙虾\preinit.py
echo Server crashed, restarting in 5 seconds...
timeout /t 5
goto loop
