---
title: "Upgrade nx repo to react 18"
date: 2022-07-21T17:21:25+08:00
draft: true
---
- 更新@nrwl/react 到14會自動更新React到18
- https://nx.dev/guides/react-18#react-18-migration
- npx 指令：暫時使用，不global下載
- 更新nx，
    - `npx nx migrate latest `: 尋找更新的dependencies，還沒更新任何東西
        - 更新package.json 
        - 產生migrations.json
    - `npx nx migrate --run-migrations`: 開始執行下載，包含npm i
https://nx.dev/using-nx/updating-nx

- 更新chakra-ui
    - 查版本https://www.npmjs.com/package/@chakra-ui/react
        - https://chakra-ui.com/getting-started/migration
    - 改package.json到最新版 （目前2.2.4)
- 更新 Typescript
    - 查版本https://www.npmjs.com/package/typescript
    - 改package.json到最新版 （目前4.7.4)
- 檢查
    - npx nx run app:lint
    - npx nx run app:build:production
- 遇到問題
    - chakra裡面的csstype下npm i沒有被下載到
        - @chakra-ui/styled-system/node_modules/csstype
        - 改path: @chakra-ui/styled-system/node_modules/csstype ~> csstype
    - Warning: ReactDOM.hydrate is no longer supported in React 18
        - next在版本12.1.7有解掉，
            - https://github.com/vercel/next.js/issues/37378