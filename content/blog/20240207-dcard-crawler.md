---
title: 20240207-dcard-crawler
date: 2024-02-07
tags:
  - web_scraping
  - dcard
  - 反爬蟲
  - selenium
  - jieba
  - playright
  - DrissionPage
updated: 2024-02-07
---
https://github.com/alanhc/dcard_crawler.git

## Backgorund
有些反爬蟲會針對selenium偵測，因此使用 undetected-chromedriver

先去 chrome://settings/help 確定chrome是最新版本

## playwright 比 seleium 快

## DrissionPage vs playwright 
- DrissionPage 比較快
	- 0.844653 秒  vs 3.023822 秒
- 不會觸發一些反爬蟲措施

## 斷詞
https://www.fontpalace.com/font-download/SimHei/
### 文字雲
![](https://i.imgur.com/3nHi6Of.png)

## ckip-transformers


### 文字雲
![](https://i.imgur.com/O3dfcbd.png)

## jieba 比較 ckip-transformers
| ![](https://i.imgur.com/O3dfcbd.png) | ![](https://i.imgur.com/YnFXpM6.png)<br> |
| ---- | ---- |
| ckip-transformers | jieba |

## Result
### 爬蟲時間：DrissionPage < Playwright < seleium
### 段詞： ckip-transformers > jieba
## Ref
- [# jieba分词过滤停顿词、标点符号及统计词频](https://zhuanlan.zhihu.com/p/39437488)
- https://github.com/alanhc/dcard_crawler.git
- https://github.com/ckiplab/ckip-transformers
- https://www.w3schools.com/cssref/css_selectors.php
https://www.zenrows.com/blog/playwright-scraping#step-five-taking-screenshots-with-playwright