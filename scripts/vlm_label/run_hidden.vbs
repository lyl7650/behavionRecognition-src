' Hidden launcher: runs watchdog.bat with no visible console window.
' Run via explorer.exe (or wscript.exe). Safe to trigger repeatedly -
' run_overnight holds a PID lock and exits if another instance is alive.
CreateObject("WScript.Shell").Run """E:\ActionRecognition\behavionRecognition-src\scripts\vlm_label\watchdog.bat""", 0, False
