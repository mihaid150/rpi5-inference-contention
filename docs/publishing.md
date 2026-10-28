# Publishing this branch

The dedicated branch is `release/rpi5-inference-document-updates`. It has its own root
commit so unrelated project files and earlier history are not reachable from
this branch. Publish only this branch, not every branch or the entire local
repository directory.

From either worktree in the original repository:

```bash
git push YOUR_REMOTE release/rpi5-inference-document-updates
```

For a new, empty personal repository where this should be the main branch:

```bash
git push YOUR_REPOSITORY_URL release/rpi5-inference-document-updates:main
```

Do not use `--mirror`, `--all` or force-push over an existing main branch.
Model weights, prepared inputs and run outputs remain external to Git.
The `.gitignore` and `.dockerignore` exclude local workspaces and generated data;
the tracked source assets are small fixed inputs required by the experiment.

A reviewer can clone only this branch:

```bash
git clone --single-branch --branch release/rpi5-inference-document-updates YOUR_REPOSITORY_URL
```

If you published it as `main`, clone that branch instead.
