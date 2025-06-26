---
title: Paste Image in VSCode
date: 2025-06-26
tags:
  - vscode
updated: 2025-06-26
draft: false
up:
---
可以藉由設定 `{專案_ROOT}/.vscode/settings.json`

```json
{
    "markdown.copyFiles.destination": {
        "/docs/**/*": "images/${documentBaseName}/"
    }
}
```
他會自動將所有 ``/docs/`` 底下的圖片複製到 ``images/${documentBaseName}/`` 底下。