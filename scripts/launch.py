import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time


def manifest(image, assets, output, node, name):
    mounts = [
        ("input", str(assets), "/input"),
        ("models", str(assets / "models"), "/models"),
        ("output", str(output), "/output"),
    ]
    return dict(
        apiVersion="batch/v1",
        kind="Job",
        metadata=dict(name=name),
        spec=dict(
            backoffLimit=0,
            activeDeadlineSeconds=259200,
            ttlSecondsAfterFinished=3600,
            template=dict(
                metadata=dict(
                    labels={"app.kubernetes.io/name": "shared-budget-evaluation"}
                ),
                spec=dict(
                    nodeSelector={"kubernetes.io/hostname": node},
                    restartPolicy="Never",
                    automountServiceAccountToken=False,
                    securityContext=dict(
                        runAsUser=os.getuid(),
                        runAsGroup=os.getgid(),
                        runAsNonRoot=True,
                        seccompProfile=dict(type="RuntimeDefault"),
                    ),
                    containers=[
                        dict(
                            name="evaluation",
                            image=image,
                            resources=dict(
                                requests=dict(cpu="4", memory="12Gi"),
                                limits=dict(cpu="4", memory="12Gi"),
                            ),
                            securityContext=dict(
                                allowPrivilegeEscalation=False,
                                readOnlyRootFilesystem=True,
                                capabilities=dict(drop=["ALL"]),
                            ),
                            volumeMounts=[
                                dict(name=n, mountPath=m, readOnly=n != "output")
                                for n, p, m in mounts
                            ]
                            + [dict(name="tmp", mountPath="/tmp")],
                        )
                    ],
                    volumes=[
                        dict(name=n, hostPath=dict(path=p, type="Directory"))
                        for n, p, m in mounts
                    ]
                    + [dict(name="tmp", emptyDir=dict(sizeLimit="512Mi"))],
                ),
            ),
        ),
    )


def command(image, assets, output, name):
    return [
        "docker",
        "run",
        "--detach",
        "--rm",
        "--name",
        name,
        "--cpus",
        "4",
        "--cpuset-cpus",
        "0-3",
        "--memory",
        "12g",
        "--memory-swap",
        "12g",
        "--network",
        "none",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--tmpfs",
        "/tmp:rw,size=536870912",
        "--mount",
        f"type=bind,src={assets},dst=/input,readonly",
        "--mount",
        f'type=bind,src={assets / "models"},dst=/models,readonly',
        "--mount",
        f"type=bind,src={output},dst=/output",
        image,
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=["docker", "kubernetes"], default="docker")
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", default="shared-budget-evaluation:local")
    parser.add_argument("--node")
    parser.add_argument("--namespace", default="default")
    parser.add_argument("--name", default="evaluation-" + time.strftime("%Y%m%d%H%M%S"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch("[a-z0-9]([-a-z0-9]{0,60}[a-z0-9])?", args.name):
        parser.error("Name must be a Kubernetes-compatible identifier")
    assets, output = args.assets.resolve(), args.output.resolve()
    if args.engine == "kubernetes" and not args.node:
        parser.error("--node is required for Kubernetes hostPath placement")
    if args.engine == "docker":
        payload = command(args.image, assets, output, args.name)
        if args.dry_run:
            print(shlex.join(payload))
            return
    else:
        payload = manifest(args.image, assets, output, args.node, args.name)
        if args.dry_run:
            print(json.dumps(payload, indent=2))
            return
    if os.getuid() == 0:
        parser.error("Run as a non-root user with access to Docker or kubectl")
    for name in ("assets-manifest.json", "models/family/Q4_K_M.gguf"):
        if not (assets / name).is_file():
            parser.error(f"Missing prepared asset: {name}")
    if output.exists():
        parser.error("Output path already exists; select a new attempt directory")
    output.mkdir(parents=True)
    if args.engine == "docker":
        subprocess.run(payload, check=True)
    else:
        subprocess.run(
            ["kubectl", "-n", args.namespace, "create", "-f", "-"],
            input=json.dumps(payload),
            text=True,
            check=True,
        )
    print(f'Status: {output / "status.json"}')
    print(f'Completion marker: {output / "DONE"}')


if __name__ == "__main__":
    main()
