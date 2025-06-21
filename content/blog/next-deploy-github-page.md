---
title: "Next deploy github page"
date: 2022-07-15T00:36:14+08:00
draft: true
---
1. 在project root新增 [`.github/workflows/deploy.yml`](https://gist.github.com/alanhc/53923b66f7011b55b9ae920936688607)

2. next.config.js 新增
    - `const isProd = process.env.NODE_ENV === 'production'`
    - 
        ```
        module.exports = {
            ...
            assetPrefix: isProd ? '/j-p1/' : ''
        }
        ```
    - example
        ```js
        /** @type {import('next').NextConfig} */
        //
        const isProd = process.env.NODE_ENV === 'production'

        const nextConfig = {
          reactStrictMode: true,  
          assetPrefix: isProd ? '/j-p1/' : ''
        }

        module.exports = nextConfig

        ```
3. 設定 `package.json`
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start",
    "lint": "next lint",
    "export": "next export"
  },

FAQ
- Action問題、ssh
    - https://github.com/peaceiris/actions-gh-pages