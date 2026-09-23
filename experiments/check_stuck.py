"""Check if the process is still running on VM and get resource usage."""
import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('192.168.102.129', 22, 'quick404', 'quick404')

cmd = "ps aux | grep -i 'cli.py\|python3' | grep -v grep | head -10"
stdin, stdout, stderr = ssh.exec_command(cmd)
out = stdout.read().decode().strip()
if out:
    print("=== Running python processes ===")
    print(out)
else:
    print("No python processes found (may have completed or crashed)")

# Also check if the output file was created
cmd2 = "ls -la ~/网络流量分析项目/nemesys-gnn4id/test_dhcp_fixed.txt 2>/dev/null || echo 'not found'"
stdin, stdout, stderr = ssh.exec_command(cmd2)
print(f"\nOutput file: {stdout.read().decode().strip()}")

# Check last lines of any partial output
cmd3 = "cat ~/网络流量分析项目/nemesys-gnn4id/test_dhcp_fixed.txt 2>/dev/null | tail -5 || echo 'no file'"
stdin, stdout, stderr = ssh.exec_command(cmd3)
out = stdout.read().decode().strip()
if out and out != 'no file':
    print(f"\n=== Partial output ===")
    print(out)

ssh.close()
