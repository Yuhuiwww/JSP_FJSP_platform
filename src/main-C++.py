import subprocess
command = [
    'C++exe\\GA_JSP.exe',
    '-inputString='
]
pi = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
for i in iter(pi.stdout.readline, 'b'):
    print(i.decode('gbk'))
