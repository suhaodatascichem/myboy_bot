import subprocess

powershell_script = """
$processes = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'main.py' }
foreach ($p in $processes) { Write-Output $p.ProcessId }
"""
p = subprocess.Popen(["powershell", "-NoProfile", "-Command", powershell_script], stdout=subprocess.PIPE, text=True)
output = p.communicate()[0]

for line in output.strip().split('\n'):
    pid = line.strip()
    if pid.isdigit():
        print(f"Killing PID: {pid}")
        subprocess.call(f"taskkill /F /PID {pid}", shell=True)
