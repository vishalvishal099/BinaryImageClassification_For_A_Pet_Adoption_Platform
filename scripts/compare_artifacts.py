#!/usr/bin/env python3
"""
Artifact Comparison Script & API

Compares locally running container artifact with GitHub Container Registry latest artifact.
Provides both CLI and API endpoints for version comparison.
"""

import json
import subprocess
import sys
from datetime import datetime
from typing import Any, Dict, Optional

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Configuration
GITHUB_REGISTRY = "ghcr.io"
GITHUB_OWNER = "vishalvishal099"
GITHUB_REPO = "binaryimageclassification_for_a_pet_adoption_platform"
LOCAL_CONTAINER_NAME = "cats-dogs-api"
LOCAL_IMAGE_NAME = "cats-dogs-classifier"

app = FastAPI(
    title="Artifact Comparison API",
    description="Compare local running artifact with GitHub Container Registry",
    version="1.0.0",
)


class ArtifactInfo(BaseModel):
    """Artifact information model."""
    source: str
    image_name: str
    image_id: str
    tag: str
    created: Optional[str] = None
    size: Optional[str] = None
    digest: Optional[str] = None
    status: str


class ComparisonResult(BaseModel):
    """Comparison result model."""
    timestamp: str
    local_artifact: ArtifactInfo
    registry_artifact: ArtifactInfo
    is_same_version: bool
    needs_update: bool
    comparison_details: Dict[str, Any]


def run_command(cmd: list) -> tuple:
    """Run a shell command and return output."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )
        return result.stdout.strip(), result.returncode
    except subprocess.TimeoutExpired:
        return "Command timed out", 1
    except Exception as e:
        return str(e), 1


def get_local_artifact_info() -> Dict[str, Any]:
    """Get information about the locally running container artifact."""
    
    # Check if container is running
    stdout, code = run_command(["podman", "ps", "-q", "-f", f"name={LOCAL_CONTAINER_NAME}"])
    if not stdout:
        return {
            "source": "local",
            "image_name": "N/A",
            "image_id": "N/A",
            "tag": "N/A",
            "created": None,
            "size": None,
            "digest": None,
            "status": "not_running"
        }
    
    # Get container details
    container_id = stdout.strip()
    
    # Get image ID from container
    image_id, _ = run_command([
        "podman", "inspect", LOCAL_CONTAINER_NAME,
        "--format", "{{.Image}}"
    ])
    
    # Get image name
    image_name, _ = run_command([
        "podman", "inspect", LOCAL_CONTAINER_NAME,
        "--format", "{{.ImageName}}"
    ])
    
    # Get container created time
    created, _ = run_command([
        "podman", "inspect", LOCAL_CONTAINER_NAME,
        "--format", "{{.Created}}"
    ])
    
    # Get image size
    size, _ = run_command([
        "podman", "images", LOCAL_IMAGE_NAME,
        "--format", "{{.Size}}"
    ])
    
    # Get image digest if available
    digest, _ = run_command([
        "podman", "images", LOCAL_IMAGE_NAME,
        "--format", "{{.Digest}}"
    ])
    
    # Extract tag from image name
    tag = "latest"
    if ":" in image_name:
        tag = image_name.split(":")[-1]
    
    return {
        "source": "local",
        "image_name": image_name,
        "image_id": image_id[:12] if image_id else "N/A",
        "full_image_id": image_id,
        "tag": tag,
        "created": created,
        "size": size,
        "digest": digest if digest and digest != "<none>" else None,
        "status": "running",
        "container_id": container_id[:12]
    }


def get_registry_artifact_info() -> Dict[str, Any]:
    """Get information about the latest artifact from GitHub Container Registry."""
    
    registry_image = f"{GITHUB_REGISTRY}/{GITHUB_OWNER}/{GITHUB_REPO}"
    
    # Try to get remote image info using skopeo or podman
    # First, try to inspect without pulling
    manifest_cmd = [
        "podman", "manifest", "inspect",
        f"{registry_image}:latest"
    ]
    manifest, code = run_command(manifest_cmd)
    
    if code != 0:
        # Try with skopeo
        skopeo_cmd = [
            "skopeo", "inspect",
            f"docker://{registry_image}:latest"
        ]
        manifest, code = run_command(skopeo_cmd)
    
    if code == 0:
        try:
            manifest_data = json.loads(manifest)
            digest = manifest_data.get("Digest", manifest_data.get("digest", "N/A"))
            created = manifest_data.get("Created", manifest_data.get("created", "N/A"))
            
            return {
                "source": "github_registry",
                "image_name": f"{registry_image}:latest",
                "image_id": digest[:19] if digest else "N/A",
                "tag": "latest",
                "created": created,
                "size": "N/A",
                "digest": digest,
                "status": "available"
            }
        except json.JSONDecodeError:
            pass
    
    # Fallback: Check if we have the image locally pulled from registry
    check_cmd = [
        "podman", "images", registry_image,
        "--format", "{{.ID}}\t{{.Digest}}\t{{.Created}}\t{{.Size}}"
    ]
    output, code = run_command(check_cmd)
    
    if output and code == 0:
        parts = output.split("\t")
        return {
            "source": "github_registry",
            "image_name": f"{registry_image}:latest",
            "image_id": parts[0] if len(parts) > 0 else "N/A",
            "tag": "latest",
            "created": parts[2] if len(parts) > 2 else "N/A",
            "size": parts[3] if len(parts) > 3 else "N/A",
            "digest": parts[1] if len(parts) > 1 else None,
            "status": "pulled_locally"
        }
    
    # Check GitHub API for package info
    try:
        # Note: This requires authentication for private packages
        api_url = f"https://api.github.com/users/{GITHUB_OWNER}/packages/container/{GITHUB_REPO}/versions"
        headers = {"Accept": "application/vnd.github+json"}
        response = requests.get(api_url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            versions = response.json()
            if versions:
                latest = versions[0]
                return {
                    "source": "github_registry",
                    "image_name": f"{registry_image}:latest",
                    "image_id": str(latest.get("id", "N/A"))[:12],
                    "tag": "latest",
                    "created": latest.get("created_at", "N/A"),
                    "size": "N/A",
                    "digest": latest.get("name", None),
                    "status": "available"
                }
    except Exception:
        pass
    
    return {
        "source": "github_registry",
        "image_name": f"{registry_image}:latest",
        "image_id": "N/A",
        "tag": "latest",
        "created": None,
        "size": None,
        "digest": None,
        "status": "not_available_or_private"
    }


def compare_artifacts(local: Dict, registry: Dict) -> Dict[str, Any]:
    """Compare local and registry artifacts."""
    
    # Check if they are the same version
    is_same = False
    needs_update = True
    
    # Compare by digest if available
    if local.get("digest") and registry.get("digest"):
        is_same = local["digest"] == registry["digest"]
        needs_update = not is_same
    # Compare by image ID
    elif local.get("full_image_id") and registry.get("image_id"):
        is_same = local["full_image_id"].startswith(registry["image_id"])
        needs_update = not is_same
    
    return {
        "is_same_version": is_same,
        "needs_update": needs_update,
        "comparison_method": "digest" if (local.get("digest") and registry.get("digest")) else "image_id",
        "local_status": local.get("status"),
        "registry_status": registry.get("status"),
        "recommendation": "Up to date!" if is_same else "Consider updating to latest registry version"
    }


@app.get("/")
async def root():
    """API root."""
    return {
        "name": "Artifact Comparison API",
        "version": "1.0.0",
        "endpoints": {
            "compare": "/compare",
            "local": "/artifact/local",
            "registry": "/artifact/registry",
            "pull_latest": "/pull-latest"
        }
    }


@app.get("/artifact/local")
async def get_local():
    """Get local running artifact information."""
    return get_local_artifact_info()


@app.get("/artifact/registry")
async def get_registry():
    """Get GitHub Container Registry artifact information."""
    return get_registry_artifact_info()


@app.get("/compare", response_model=ComparisonResult)
async def compare():
    """
    Compare local running artifact with GitHub Container Registry latest artifact.
    
    Returns detailed comparison including:
    - Local artifact info (container ID, image ID, created time)
    - Registry artifact info (digest, tags, created time)
    - Whether versions match
    - Recommendation for update
    """
    local = get_local_artifact_info()
    registry = get_registry_artifact_info()
    comparison = compare_artifacts(local, registry)
    
    return ComparisonResult(
        timestamp=datetime.utcnow().isoformat(),
        local_artifact=ArtifactInfo(
            source=local["source"],
            image_name=local["image_name"],
            image_id=local["image_id"],
            tag=local["tag"],
            created=local.get("created"),
            size=local.get("size"),
            digest=local.get("digest"),
            status=local["status"]
        ),
        registry_artifact=ArtifactInfo(
            source=registry["source"],
            image_name=registry["image_name"],
            image_id=registry["image_id"],
            tag=registry["tag"],
            created=registry.get("created"),
            size=registry.get("size"),
            digest=registry.get("digest"),
            status=registry["status"]
        ),
        is_same_version=comparison["is_same_version"],
        needs_update=comparison["needs_update"],
        comparison_details=comparison
    )


@app.post("/pull-latest")
async def pull_latest():
    """Pull the latest image from GitHub Container Registry."""
    registry_image = f"{GITHUB_REGISTRY}/{GITHUB_OWNER}/{GITHUB_REPO}:latest"
    
    stdout, code = run_command(["podman", "pull", registry_image])
    
    if code != 0:
        raise HTTPException(status_code=500, detail=f"Failed to pull image: {stdout}")
    
    return {
        "status": "success",
        "message": f"Successfully pulled {registry_image}",
        "output": stdout
    }


def cli_compare():
    """CLI comparison function."""
    print("=" * 70)
    print("  ARTIFACT COMPARISON: Local vs GitHub Container Registry")
    print("=" * 70)
    
    print("\n📦 Fetching local artifact info...")
    local = get_local_artifact_info()
    
    print("\n🌐 Fetching registry artifact info...")
    registry = get_registry_artifact_info()
    
    print("\n" + "-" * 70)
    print("  LOCAL ARTIFACT")
    print("-" * 70)
    print(f"  Image:      {local['image_name']}")
    print(f"  Image ID:   {local['image_id']}")
    print(f"  Status:     {local['status']}")
    print(f"  Created:    {local.get('created', 'N/A')}")
    print(f"  Size:       {local.get('size', 'N/A')}")
    print(f"  Digest:     {local.get('digest', 'N/A')}")
    
    print("\n" + "-" * 70)
    print("  GITHUB REGISTRY ARTIFACT")
    print("-" * 70)
    print(f"  Image:      {registry['image_name']}")
    print(f"  Image ID:   {registry['image_id']}")
    print(f"  Status:     {registry['status']}")
    print(f"  Created:    {registry.get('created', 'N/A')}")
    print(f"  Digest:     {registry.get('digest', 'N/A')}")
    
    comparison = compare_artifacts(local, registry)
    
    print("\n" + "=" * 70)
    print("  COMPARISON RESULT")
    print("=" * 70)
    
    if comparison["is_same_version"]:
        print("  ✅ MATCH: Local and registry artifacts are the SAME version!")
    else:
        print("  ⚠️  DIFFERENT: Local and registry artifacts are DIFFERENT versions!")
        print(f"  📌 Recommendation: {comparison['recommendation']}")
    
    print("\n  To update local to latest:")
    print(f"  $ podman pull {GITHUB_REGISTRY}/{GITHUB_OWNER}/{GITHUB_REPO}:latest")
    print(f"  $ podman stop {LOCAL_CONTAINER_NAME}")
    print(f"  $ podman run -d --name {LOCAL_CONTAINER_NAME} -p 8000:8000 \\")
    print(f"      {GITHUB_REGISTRY}/{GITHUB_OWNER}/{GITHUB_REPO}:latest")
    print()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8001)
    else:
        cli_compare()
