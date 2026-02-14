#!/usr/bin/env python3
"""
CD Pipeline Watcher & Auto-Deployer

This script watches for GitHub Actions CI completion and automatically:
1. Pulls the latest artifact from GitHub Container Registry
2. Deploys it to local Podman
3. Provides real-time visualization of the deployment process

Usage:
    python scripts/cd_watcher.py              # Watch mode (continuous)
    python scripts/cd_watcher.py --once       # Single check and deploy
    python scripts/cd_watcher.py --dashboard  # Start web dashboard
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

# Try to import optional dependencies
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from flask import Flask, Response, jsonify, render_template_string
    from flask_cors import CORS
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False


# =============================================================================
# Configuration
# =============================================================================

class Config:
    GITHUB_OWNER = "vishalvishal099"
    GITHUB_REPO = "BinaryImageClassification_For_A_Pet_Adoption_Platform"
    REGISTRY_IMAGE = f"ghcr.io/{GITHUB_OWNER.lower()}/{GITHUB_REPO.lower()}"
    LOCAL_CONTAINER_NAME = "cats-dogs-api"
    LOCAL_PORT = 8000
    POLL_INTERVAL = 30  # seconds
    DASHBOARD_PORT = 8080


# =============================================================================
# Pipeline Status
# =============================================================================

class PipelineStage(Enum):
    IDLE = "idle"
    CHECKING_CI = "checking_ci"
    CI_RUNNING = "ci_running"
    CI_COMPLETED = "ci_completed"
    PULLING_IMAGE = "pulling_image"
    STOPPING_OLD = "stopping_old"
    STARTING_NEW = "starting_new"
    HEALTH_CHECK = "health_check"
    DEPLOYED = "deployed"
    FAILED = "failed"


@dataclass
class PipelineEvent:
    timestamp: str
    stage: str
    message: str
    status: str  # success, error, info, warning
    details: Optional[Dict] = None


@dataclass
class PipelineState:
    current_stage: PipelineStage = PipelineStage.IDLE
    events: List[PipelineEvent] = field(default_factory=list)
    last_ci_run_id: Optional[int] = None
    last_deployed_sha: Optional[str] = None
    current_image_id: Optional[str] = None
    is_watching: bool = False
    last_check: Optional[str] = None
    
    def add_event(self, stage: str, message: str, status: str = "info", details: Dict = None):
        event = PipelineEvent(
            timestamp=datetime.utcnow().isoformat(),
            stage=stage,
            message=message,
            status=status,
            details=details or {}
        )
        self.events.append(event)
        # Keep only last 100 events
        if len(self.events) > 100:
            self.events = self.events[-100:]
        return event
    
    def to_dict(self) -> Dict:
        return {
            "current_stage": self.current_stage.value,
            "events": [
                {
                    "timestamp": e.timestamp,
                    "stage": e.stage,
                    "message": e.message,
                    "status": e.status,
                    "details": e.details
                }
                for e in self.events[-20:]  # Last 20 events
            ],
            "last_ci_run_id": self.last_ci_run_id,
            "last_deployed_sha": self.last_deployed_sha,
            "current_image_id": self.current_image_id,
            "is_watching": self.is_watching,
            "last_check": self.last_check
        }


# Global state
pipeline_state = PipelineState()


# =============================================================================
# Utility Functions
# =============================================================================

def run_command(cmd: List[str], timeout: int = 120) -> tuple:
    """Run a shell command and return output."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.stdout.strip(), result.stderr.strip(), result.returncode
    except subprocess.TimeoutExpired:
        return "", "Command timed out", 1
    except Exception as e:
        return "", str(e), 1


def get_gh_token() -> Optional[str]:
    """Get GitHub token from gh CLI or environment."""
    # Try environment variable first
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        return token
    
    # Try gh CLI
    stdout, _, code = run_command(["gh", "auth", "token"])
    if code == 0 and stdout:
        return stdout.strip()
    
    return None


def print_stage(stage: str, message: str, status: str = "info"):
    """Print formatted stage message."""
    icons = {
        "info": "ℹ️ ",
        "success": "✅",
        "error": "❌",
        "warning": "⚠️ ",
        "progress": "🔄"
    }
    icon = icons.get(status, "  ")
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {icon} [{stage}] {message}")
    
    # Add to pipeline state
    pipeline_state.add_event(stage, message, status)


# =============================================================================
# GitHub Actions Integration
# =============================================================================

def check_latest_ci_run() -> Optional[Dict]:
    """Check the latest CI workflow run status."""
    pipeline_state.current_stage = PipelineStage.CHECKING_CI
    print_stage("CI", "Checking latest CI workflow run...", "progress")
    
    # Use gh CLI to get workflow runs
    stdout, stderr, code = run_command([
        "gh", "run", "list",
        "--workflow=ci.yml",
        "--limit=1",
        "--json", "databaseId,status,conclusion,headSha,createdAt,displayTitle"
    ])
    
    if code != 0:
        print_stage("CI", f"Failed to check CI status: {stderr}", "error")
        return None
    
    try:
        runs = json.loads(stdout)
        if runs:
            run = runs[0]
            pipeline_state.last_check = datetime.utcnow().isoformat()
            return {
                "id": run.get("databaseId"),
                "status": run.get("status"),
                "conclusion": run.get("conclusion"),
                "sha": run.get("headSha"),
                "created_at": run.get("createdAt"),
                "title": run.get("displayTitle")
            }
    except json.JSONDecodeError:
        print_stage("CI", "Failed to parse CI response", "error")
    
    return None


def wait_for_ci_completion(run_id: int, timeout: int = 600) -> bool:
    """Wait for a CI run to complete."""
    pipeline_state.current_stage = PipelineStage.CI_RUNNING
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        stdout, _, code = run_command([
            "gh", "run", "view", str(run_id),
            "--json", "status,conclusion"
        ])
        
        if code == 0:
            try:
                data = json.loads(stdout)
                status = data.get("status")
                conclusion = data.get("conclusion")
                
                if status == "completed":
                    if conclusion == "success":
                        print_stage("CI", f"CI run {run_id} completed successfully!", "success")
                        pipeline_state.current_stage = PipelineStage.CI_COMPLETED
                        return True
                    else:
                        print_stage("CI", f"CI run {run_id} failed: {conclusion}", "error")
                        pipeline_state.current_stage = PipelineStage.FAILED
                        return False
                else:
                    print_stage("CI", f"CI run {run_id} status: {status}...", "progress")
            except json.JSONDecodeError:
                pass
        
        time.sleep(10)
    
    print_stage("CI", f"Timeout waiting for CI run {run_id}", "error")
    return False


# =============================================================================
# Container Deployment
# =============================================================================

def pull_latest_image() -> bool:
    """Pull the latest image from GitHub Container Registry."""
    pipeline_state.current_stage = PipelineStage.PULLING_IMAGE
    print_stage("PULL", f"Pulling latest image: {Config.REGISTRY_IMAGE}:latest", "progress")
    
    # Login to ghcr.io first
    token = get_gh_token()
    if token:
        login_cmd = f"echo {token} | podman login ghcr.io -u {Config.GITHUB_OWNER} --password-stdin"
        os.system(login_cmd + " 2>/dev/null")
    
    stdout, stderr, code = run_command(
        ["podman", "pull", f"{Config.REGISTRY_IMAGE}:latest"],
        timeout=300
    )
    
    if code != 0:
        print_stage("PULL", f"Failed to pull image: {stderr}", "error")
        pipeline_state.current_stage = PipelineStage.FAILED
        return False
    
    # Get the new image ID
    img_id, _, _ = run_command([
        "podman", "images", Config.REGISTRY_IMAGE,
        "--format", "{{.ID}}"
    ])
    
    if img_id:
        pipeline_state.current_image_id = img_id[:12]
        print_stage("PULL", f"Pulled image ID: {img_id[:12]}", "success")
    
    return True


def stop_old_container() -> bool:
    """Stop and remove the old container."""
    pipeline_state.current_stage = PipelineStage.STOPPING_OLD
    print_stage("STOP", f"Stopping old container: {Config.LOCAL_CONTAINER_NAME}", "progress")
    
    # Stop container
    run_command(["podman", "stop", Config.LOCAL_CONTAINER_NAME])
    
    # Remove container
    run_command(["podman", "rm", Config.LOCAL_CONTAINER_NAME])
    
    print_stage("STOP", "Old container stopped and removed", "success")
    return True


def start_new_container(git_sha: str = None) -> bool:
    """Start a new container with the latest image."""
    pipeline_state.current_stage = PipelineStage.STARTING_NEW
    print_stage("START", "Starting new container...", "progress")
    
    # Get image ID for environment variable
    img_id, _, _ = run_command([
        "podman", "images", f"{Config.REGISTRY_IMAGE}:latest",
        "--format", "{{.ID}}"
    ])
    
    env_vars = [
        "-e", f"LOCAL_IMAGE_ID={img_id[:12] if img_id else 'unknown'}",
        "-e", f"CONTAINER_IMAGE={Config.REGISTRY_IMAGE}:latest",
        "-e", f"GIT_SHA={git_sha or 'unknown'}",
        "-e", f"BUILD_TIMESTAMP={datetime.utcnow().isoformat()}",
    ]
    
    cmd = [
        "podman", "run", "-d",
        "--name", Config.LOCAL_CONTAINER_NAME,
        "-p", f"{Config.LOCAL_PORT}:8000",
    ] + env_vars + [f"{Config.REGISTRY_IMAGE}:latest"]
    
    stdout, stderr, code = run_command(cmd)
    
    if code != 0:
        print_stage("START", f"Failed to start container: {stderr}", "error")
        pipeline_state.current_stage = PipelineStage.FAILED
        return False
    
    print_stage("START", f"Container started: {stdout[:12]}", "success")
    return True


def health_check(retries: int = 10, delay: int = 3) -> bool:
    """Perform health check on the deployed container."""
    pipeline_state.current_stage = PipelineStage.HEALTH_CHECK
    print_stage("HEALTH", "Performing health check...", "progress")
    
    url = f"http://localhost:{Config.LOCAL_PORT}/health"
    
    for i in range(retries):
        try:
            if REQUESTS_AVAILABLE:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    if data.get("status") == "healthy":
                        print_stage("HEALTH", "Health check passed!", "success")
                        return True
            else:
                # Fallback to curl
                stdout, _, code = run_command(["curl", "-s", "-f", url])
                if code == 0:
                    print_stage("HEALTH", "Health check passed!", "success")
                    return True
        except Exception:
            pass
        
        print_stage("HEALTH", f"Health check attempt {i+1}/{retries}...", "progress")
        time.sleep(delay)
    
    print_stage("HEALTH", "Health check failed after retries", "error")
    pipeline_state.current_stage = PipelineStage.FAILED
    return False


# =============================================================================
# Main Deployment Pipeline
# =============================================================================

def deploy_latest(git_sha: str = None) -> bool:
    """Execute the full deployment pipeline."""
    print("\n" + "=" * 60)
    print("  🚀 CD PIPELINE: Auto-Deployment")
    print("=" * 60 + "\n")
    
    steps = [
        ("Pull Image", lambda: pull_latest_image()),
        ("Stop Old Container", lambda: stop_old_container()),
        ("Start New Container", lambda: start_new_container(git_sha)),
        ("Health Check", lambda: health_check()),
    ]
    
    for step_name, step_func in steps:
        if not step_func():
            print_stage("DEPLOY", f"Deployment failed at: {step_name}", "error")
            return False
    
    pipeline_state.current_stage = PipelineStage.DEPLOYED
    pipeline_state.last_deployed_sha = git_sha
    
    print("\n" + "=" * 60)
    print("  ✅ DEPLOYMENT SUCCESSFUL!")
    print("=" * 60)
    print(f"\n  📍 API available at: http://localhost:{Config.LOCAL_PORT}")
    print(f"  📍 Docs available at: http://localhost:{Config.LOCAL_PORT}/docs")
    print(f"  📍 Artifacts API: http://localhost:{Config.LOCAL_PORT}/artifacts/compare")
    print()
    
    return True


def watch_and_deploy():
    """Watch for CI completions and auto-deploy."""
    pipeline_state.is_watching = True
    print("\n" + "=" * 60)
    print("  👁️  CD WATCHER: Monitoring for CI completions")
    print("=" * 60)
    print(f"\n  Polling every {Config.POLL_INTERVAL} seconds...")
    print("  Press Ctrl+C to stop\n")
    
    last_successful_run_id = None
    
    while pipeline_state.is_watching:
        try:
            # Check latest CI run
            ci_run = check_latest_ci_run()
            
            if ci_run:
                run_id = ci_run.get("id")
                status = ci_run.get("status")
                conclusion = ci_run.get("conclusion")
                sha = ci_run.get("sha", "")[:7]
                
                print_stage("WATCH", f"Latest CI: #{run_id} - {status} ({conclusion or 'pending'})", "info")
                
                # If CI is still running, wait for it
                if status == "in_progress" or status == "queued":
                    pipeline_state.last_ci_run_id = run_id
                    if wait_for_ci_completion(run_id):
                        # CI completed successfully, deploy!
                        if run_id != last_successful_run_id:
                            deploy_latest(sha)
                            last_successful_run_id = run_id
                
                # If CI completed successfully and we haven't deployed this run
                elif status == "completed" and conclusion == "success":
                    if run_id != last_successful_run_id:
                        print_stage("WATCH", f"New successful CI run detected: #{run_id}", "success")
                        deploy_latest(sha)
                        last_successful_run_id = run_id
                    else:
                        print_stage("WATCH", "Already deployed latest successful run", "info")
            
            # Wait before next check
            pipeline_state.current_stage = PipelineStage.IDLE
            time.sleep(Config.POLL_INTERVAL)
            
        except KeyboardInterrupt:
            print_stage("WATCH", "Stopping watcher...", "warning")
            pipeline_state.is_watching = False
            break
        except Exception as e:
            print_stage("WATCH", f"Error: {e}", "error")
            time.sleep(Config.POLL_INTERVAL)


# =============================================================================
# Web Dashboard
# =============================================================================

DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>CD Pipeline Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #fff;
            min-height: 100vh;
            padding: 20px;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        h1 { 
            text-align: center; 
            margin-bottom: 30px;
            font-size: 2.5rem;
            background: linear-gradient(90deg, #00d2ff, #3a7bd5);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .grid { display: grid; grid-template-columns: 1fr 2fr; gap: 20px; }
        .card {
            background: rgba(255,255,255,0.1);
            border-radius: 15px;
            padding: 20px;
            backdrop-filter: blur(10px);
        }
        .card h2 { margin-bottom: 15px; color: #00d2ff; font-size: 1.2rem; }
        .status-badge {
            display: inline-block;
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 1.1rem;
            margin-bottom: 15px;
        }
        .status-idle { background: #444; }
        .status-progress { background: #f39c12; animation: pulse 1.5s infinite; }
        .status-success { background: #27ae60; }
        .status-error { background: #e74c3c; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.7; } }
        .pipeline-stages {
            display: flex;
            justify-content: space-between;
            margin: 20px 0;
            flex-wrap: wrap;
        }
        .stage {
            text-align: center;
            padding: 10px;
            flex: 1;
            min-width: 100px;
        }
        .stage-icon {
            width: 50px;
            height: 50px;
            border-radius: 50%;
            background: #444;
            margin: 0 auto 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.5rem;
        }
        .stage.active .stage-icon { background: #f39c12; animation: pulse 1s infinite; }
        .stage.completed .stage-icon { background: #27ae60; }
        .stage.failed .stage-icon { background: #e74c3c; }
        .stage-label { font-size: 0.8rem; color: #aaa; }
        .events {
            max-height: 400px;
            overflow-y: auto;
        }
        .event {
            padding: 10px;
            border-left: 3px solid #444;
            margin-bottom: 10px;
            background: rgba(0,0,0,0.2);
            border-radius: 0 8px 8px 0;
        }
        .event.success { border-color: #27ae60; }
        .event.error { border-color: #e74c3c; }
        .event.warning { border-color: #f39c12; }
        .event.progress { border-color: #3498db; }
        .event-time { font-size: 0.75rem; color: #888; }
        .event-stage { font-weight: bold; color: #00d2ff; }
        .event-message { margin-top: 5px; }
        .info-row { 
            display: flex; 
            justify-content: space-between; 
            padding: 10px 0; 
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        .info-label { color: #888; }
        .info-value { font-family: monospace; color: #00d2ff; }
        .btn {
            background: linear-gradient(90deg, #00d2ff, #3a7bd5);
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            color: #fff;
            font-weight: bold;
            cursor: pointer;
            margin: 5px;
            transition: transform 0.2s;
        }
        .btn:hover { transform: scale(1.05); }
        .btn:disabled { opacity: 0.5; cursor: not-allowed; }
        .actions { margin-top: 20px; text-align: center; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 CD Pipeline Dashboard</h1>
        
        <div class="pipeline-stages" id="stages">
            <div class="stage" data-stage="checking_ci">
                <div class="stage-icon">🔍</div>
                <div class="stage-label">Check CI</div>
            </div>
            <div class="stage" data-stage="ci_running">
                <div class="stage-icon">⚙️</div>
                <div class="stage-label">CI Running</div>
            </div>
            <div class="stage" data-stage="pulling_image">
                <div class="stage-icon">📦</div>
                <div class="stage-label">Pull Image</div>
            </div>
            <div class="stage" data-stage="stopping_old">
                <div class="stage-icon">🛑</div>
                <div class="stage-label">Stop Old</div>
            </div>
            <div class="stage" data-stage="starting_new">
                <div class="stage-icon">▶️</div>
                <div class="stage-label">Start New</div>
            </div>
            <div class="stage" data-stage="health_check">
                <div class="stage-icon">💚</div>
                <div class="stage-label">Health</div>
            </div>
            <div class="stage" data-stage="deployed">
                <div class="stage-icon">✅</div>
                <div class="stage-label">Deployed</div>
            </div>
        </div>
        
        <div class="grid">
            <div class="card">
                <h2>📊 Current Status</h2>
                <div class="status-badge" id="status-badge">IDLE</div>
                
                <div class="info-row">
                    <span class="info-label">Watching:</span>
                    <span class="info-value" id="watching">-</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Last CI Run:</span>
                    <span class="info-value" id="last-ci">-</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Deployed SHA:</span>
                    <span class="info-value" id="deployed-sha">-</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Image ID:</span>
                    <span class="info-value" id="image-id">-</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Last Check:</span>
                    <span class="info-value" id="last-check">-</span>
                </div>
                
                <div class="actions">
                    <button class="btn" onclick="deployNow()">🚀 Deploy Now</button>
                    <button class="btn" onclick="refreshStatus()">🔄 Refresh</button>
                </div>
            </div>
            
            <div class="card">
                <h2>📜 Event Log</h2>
                <div class="events" id="events"></div>
            </div>
        </div>
    </div>
    
    <script>
        const stageOrder = ['idle', 'checking_ci', 'ci_running', 'ci_completed', 'pulling_image', 'stopping_old', 'starting_new', 'health_check', 'deployed'];
        
        function updateDashboard(data) {
            // Update status badge
            const badge = document.getElementById('status-badge');
            const stage = data.current_stage;
            badge.textContent = stage.toUpperCase().replace('_', ' ');
            badge.className = 'status-badge';
            if (stage === 'deployed') badge.classList.add('status-success');
            else if (stage === 'failed') badge.classList.add('status-error');
            else if (stage === 'idle') badge.classList.add('status-idle');
            else badge.classList.add('status-progress');
            
            // Update info
            document.getElementById('watching').textContent = data.is_watching ? '✅ Yes' : '❌ No';
            document.getElementById('last-ci').textContent = data.last_ci_run_id || '-';
            document.getElementById('deployed-sha').textContent = data.last_deployed_sha || '-';
            document.getElementById('image-id').textContent = data.current_image_id || '-';
            document.getElementById('last-check').textContent = data.last_check ? new Date(data.last_check).toLocaleTimeString() : '-';
            
            // Update stages
            const currentIdx = stageOrder.indexOf(stage);
            document.querySelectorAll('.stage').forEach((el, idx) => {
                const stageKey = el.dataset.stage;
                const stageIdx = stageOrder.indexOf(stageKey);
                el.classList.remove('active', 'completed', 'failed');
                if (stage === 'failed' && stageIdx === currentIdx) {
                    el.classList.add('failed');
                } else if (stageIdx < currentIdx || stage === 'deployed') {
                    el.classList.add('completed');
                } else if (stageIdx === currentIdx) {
                    el.classList.add('active');
                }
            });
            
            // Update events
            const eventsDiv = document.getElementById('events');
            eventsDiv.innerHTML = data.events.reverse().map(e => `
                <div class="event ${e.status}">
                    <span class="event-time">${new Date(e.timestamp).toLocaleTimeString()}</span>
                    <span class="event-stage">[${e.stage}]</span>
                    <div class="event-message">${e.message}</div>
                </div>
            `).join('');
        }
        
        async function refreshStatus() {
            try {
                const res = await fetch('/api/status');
                const data = await res.json();
                updateDashboard(data);
            } catch (e) {
                console.error('Failed to fetch status:', e);
            }
        }
        
        async function deployNow() {
            try {
                const res = await fetch('/api/deploy', { method: 'POST' });
                const data = await res.json();
                alert(data.message || 'Deployment triggered!');
                refreshStatus();
            } catch (e) {
                alert('Failed to trigger deployment');
            }
        }
        
        // Auto-refresh every 2 seconds
        setInterval(refreshStatus, 2000);
        refreshStatus();
    </script>
</body>
</html>
"""


def create_dashboard_app():
    """Create Flask dashboard application."""
    if not FLASK_AVAILABLE:
        print("Flask not available. Install with: pip install flask flask-cors")
        return None
    
    app = Flask(__name__)
    CORS(app)
    
    @app.route('/')
    def dashboard():
        return render_template_string(DASHBOARD_HTML)
    
    @app.route('/api/status')
    def api_status():
        return jsonify(pipeline_state.to_dict())
    
    @app.route('/api/deploy', methods=['POST'])
    def api_deploy():
        def deploy_thread():
            ci_run = check_latest_ci_run()
            sha = ci_run.get("sha", "")[:7] if ci_run else None
            deploy_latest(sha)
        
        thread = threading.Thread(target=deploy_thread)
        thread.start()
        return jsonify({"message": "Deployment triggered", "status": "started"})
    
    @app.route('/api/events')
    def api_events():
        def generate():
            last_count = 0
            while True:
                if len(pipeline_state.events) > last_count:
                    for event in pipeline_state.events[last_count:]:
                        yield f"data: {json.dumps({'event': event.__dict__})}\n\n"
                    last_count = len(pipeline_state.events)
                time.sleep(1)
        return Response(generate(), mimetype='text/event-stream')
    
    return app


def start_dashboard():
    """Start the web dashboard."""
    app = create_dashboard_app()
    if app:
        print(f"\n🌐 Dashboard available at: http://localhost:{Config.DASHBOARD_PORT}\n")
        
        # Start watcher in background thread
        watcher_thread = threading.Thread(target=watch_and_deploy, daemon=True)
        watcher_thread.start()
        
        # Start Flask app
        app.run(host='0.0.0.0', port=Config.DASHBOARD_PORT, debug=False)


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="CD Pipeline Watcher & Auto-Deployer")
    parser.add_argument('--once', action='store_true', help='Single check and deploy')
    parser.add_argument('--dashboard', action='store_true', help='Start web dashboard')
    parser.add_argument('--deploy', action='store_true', help='Force deploy latest')
    args = parser.parse_args()
    
    if args.dashboard:
        start_dashboard()
    elif args.deploy or args.once:
        ci_run = check_latest_ci_run()
        if ci_run and ci_run.get("conclusion") == "success":
            sha = ci_run.get("sha", "")[:7]
            deploy_latest(sha)
        elif ci_run and ci_run.get("status") == "in_progress":
            if wait_for_ci_completion(ci_run.get("id")):
                deploy_latest(ci_run.get("sha", "")[:7])
        else:
            print_stage("DEPLOY", "No successful CI run found", "warning")
            if args.deploy:
                deploy_latest("manual")
    else:
        watch_and_deploy()


if __name__ == "__main__":
    main()
