# gpano

CLI tools for downloading Google Street View panoramas

### Installation

Basic installation (only `load` utility):

```
pipx install .
```

For using `sample` utility, `torch` need to be installed, which can be done using this command:

```
pipx install .[sampling]
```

For using torch with CUDA, inject torch with CUDA-support manually:

```
pipx inject gpano torch --index-url https://download.pytorch.org/whl/cu126
```

For installing package in editable mode, use `-e` flag for `pipx`:

```
pipx install -e .
```
