#!/usr/bin/env python3
"""
Sandbox Manager - Docker-based safe execution environment
Provides isolated shell command execution and Python code running
"""

import docker
import tempfile
import os
import uuid
from typing import Dict, Any
import time
import logging

class DockerSandbox:
    """Docker-based sandbox for safe code and shell execution"""
    
    def __init__(self, workspace_dir: str = "workspace"):
        self.workspace_dir = workspace_dir
        self.client = None
        self.container = None
        self.container_id = None
        self.setup_docker()
    
    def setup_docker(self):
        """Initialize Docker client and verify connection"""
        try:
            self.client = docker.from_env()
            # Test Docker connection
            self.client.ping()
            print("Docker connection established")
        except Exception as e:
            print(f"Docker setup failed: {e}")
            self.client = None
    
    def create_container(self, session_id: str) -> bool:
        """Create a new container for this session"""
        if not self.client:
            return False
        
        try:
            # Create workspace volume
            workspace_path = os.path.abspath(self.workspace_dir)
            os.makedirs(workspace_path, exist_ok=True)
            
            # Container configuration
            container_config = {
                'image': 'python:3.10-slim',
                'detach': True,
                'tty': True,
                'working_dir': '/app',
                'volumes': {
                    workspace_path: {'bind': '/app/workspace', 'mode': 'rw'}  # Direct mount
                },
                'mem_limit': '512m',
                'cpu_period': 100000,
                'cpu_quota': 50000,
                'network_disabled': False,
                'security_opt': ['no-new-privileges:true'],
                'cap_drop': ['ALL'],
                'cap_add': ['CHOWN', 'DAC_OVERRIDE', 'SETGID', 'SETUID'],
                'read_only': False,
                'user': 'root'
            }
            
            # Create and start container
            self.container = self.client.containers.run(**container_config)
            self.container_id = self.container.id[:12]
            
            print(f"Container {self.container_id} created for session {session_id}")
            
            # Install packages
            self._install_packages()
            
            # Verify workspace mount
            self._verify_workspace_mount()
            
            return True
            
        except Exception as e:
            print(f"Failed to create container: {e}")
            return False
    
    def _install_packages(self):
        """Install essential packages in the container"""
        print("Installing essential packages...")
        
        # First check if container is healthy
        try:
            health_check = self.container.exec_run('python -c "print(\'Container healthy\')"', user='root')
            if health_check.exit_code != 0:
                print("⚠ Container not healthy, skipping package installation")
                return
            print("Container health check passed")
        except Exception as e:
            print(f"Container health check failed: {e}")
            return
        
        # Install essential packages for data science
        essential_packages = 'pip install --no-cache-dir pandas numpy scikit-learn matplotlib seaborn plotly'
        try:
            print("Installing data science packages...")
            result = self.container.exec_run(essential_packages, user='root')
            if result.exit_code == 0:
                print("✅ Data science packages (pandas, numpy, scikit-learn, matplotlib, seaborn, plotly) installed")
            else:
                print(f"⚠ Essential package install failed: {result.output.decode()[:300]}...")
        except Exception as e:
            print(f"Essential package install error: {e}")
        
        print("Package installation completed")
    
    def _verify_workspace_mount(self):
        """Verify workspace is properly mounted"""
        try:
            result = self.container.exec_run('ls -la /workspace', workdir='/app/workspace')
            if result.exit_code == 0:
                print("✅ Workspace mounted successfully")
                print(result.output.decode('utf-8', errors='ignore'))
            else:
                print("⚠️ Workspace mount verification failed")
        except Exception as e:
            print(f"Workspace verification error: {e}")
    
    def execute_python_code(self, code: str, timeout: int = 120) -> Dict[str, Any]:
        """Execute Python code in container - NO local fallback"""
        if not self.container:
            return {
                'success': False,
                'output': '',
                'error': 'No Docker container available for execution',
                'exit_code': -1
            }
        
        try:
            # Create temp file in container with proper escaping
            temp_filename = f'/tmp/exec_{uuid.uuid4().hex[:8]}.py'
            
            # Write code to container using Python within the container (most reliable)
            import base64
            encoded_code = base64.b64encode(code.encode('utf-8')).decode('ascii')
            write_python_cmd = f'''python -c "
import base64
code = '{encoded_code}'
decoded = base64.b64decode(code).decode('utf-8')
with open('{temp_filename}', 'w') as f:
    f.write(decoded)
print('File written successfully')
"'''
            
            write_result = self.container.exec_run(['bash', '-c', write_python_cmd], workdir='/workspace')
            if write_result.exit_code != 0:
                return {
                    'success': False,
                    'output': '',
                    'error': f'Failed to write code to container: {write_result.output.decode()}'
                }
            
            # Execute Python file
            result = self.container.exec_run(
                f'timeout {timeout} python {temp_filename}',
                workdir='/workspace'
            )
            
            # Clean up temp file
            self.container.exec_run(f'rm -f {temp_filename}')
            
            output = result.output.decode('utf-8', errors='ignore')
            
            return {
                'success': result.exit_code == 0,
                'output': output if result.exit_code == 0 else '',
                'error': output if result.exit_code != 0 else '',
                'exit_code': result.exit_code
            }
            
        except Exception as e:
            return {
                'success': False,
                'output': '',
                'error': f'Container execution error: {str(e)}'
            }
    
    def execute_shell_command(self, command: str, timeout: int = 30) -> Dict[str, Any]:
        """Execute shell command in container - NO local fallback"""
        if not self.container:
            return {'success': False, 'output': '', 'error': 'No container available'}
        
        try:
            # Security check
            if not self._is_safe_command(command):
                return {
                    'success': False,
                    'output': '',
                    'error': f'Command blocked for security: {command}'
                }
            
            # Execute in container
            result = self.container.exec_run(
                f'timeout {timeout} bash -c "{command}"',
                workdir='/workspace',
                user='root'
            )
            
            output = result.output.decode('utf-8', errors='ignore')
            
            return {
                'success': result.exit_code == 0,
                'output': output if result.exit_code == 0 else '',
                'error': output if result.exit_code != 0 else '',
                'exit_code': result.exit_code
            }
            
        except Exception as e:
            return {
                'success': False,
                'output': '',
                'error': f'Container execution error: {str(e)}'
            }
    
    def _is_safe_command(self, command: str) -> bool:
        """Basic security check for shell commands"""
        dangerous_patterns = [
            'rm -rf /', 'format', 'fdisk', 'mkfs', '> /dev/', 'dd if=',
            'wget', 'curl', 'nc ', 'netcat', 'sudo', 'su -',
            'passwd', 'chmod 777', 'iptables', 'systemctl'
        ]
        
        command_lower = command.lower()
        return not any(pattern in command_lower for pattern in dangerous_patterns)
    
    def cleanup(self):
        """Container cleanup"""
        if self.container:
            try:
                print(f"Cleaning up container {self.container_id}")
                self.container.stop(timeout=5)
                self.container.remove()
                print(f"✅ Container {self.container_id} cleaned up")
            except Exception as e:
                print(f"⚠ Container cleanup error: {e}")
            finally:
                self.container = None
                self.container_id = None


class SandboxManager:
    """Manager for Docker sandboxes with session-based isolation"""
    
    def __init__(self, workspace_dir: str = "workspace"):
        self.workspace_dir = workspace_dir
        self.sandboxes = {}  # session_id -> DockerSandbox

    def get_sandbox(self, session_id: str) -> DockerSandbox:
        """Get or create sandbox for session"""
        if session_id not in self.sandboxes:
            sandbox = DockerSandbox(self.workspace_dir)
            if sandbox.create_container(session_id):
                self.sandboxes[session_id] = sandbox
                print(f"✅ Sandbox created for session: {session_id}")
            else:
                raise Exception(f"Failed to create Docker sandbox for session {session_id}. "
                            f"Docker may not be available or container creation failed.")
        
        return self.sandboxes[session_id]
    
    def cleanup_session(self, session_id: str):
        """Cleanup specific session"""
        if session_id in self.sandboxes:
            self.sandboxes[session_id].cleanup()
            del self.sandboxes[session_id]
    
    def cleanup_all(self):
        """Cleanup all sessions"""
        for session_id in list(self.sandboxes.keys()):
            self.cleanup_session(session_id)
