---
title: 使用Bloom Filter及web3.py來做疫苗認證 — NTU DApp 期中prototype
date: 2022-04-06
tags:
  - algorithm
  - medium
  - select
---
前言
==

剛好最近看到一篇blockchain-based的疫苗認證系統提到他們使用Bloom Filter來快速查找疫苗批號，因此就萌生了想自己來實作一下這篇的其中一個功能-使用Bloom Filter來追蹤疫苗批號，先來看一下我做了什麼👉 \[53s\]
<iframe width="560" height="315" src="https://www.youtube.com/embed/oaXrKvUwzuU?si=rniuYQAxKhlL-ynF" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
為什麼使用Bloom Filter？
==================

在區塊鏈應用裡面，如何節省storage fee(使用空間)及快速找到資料(快速搜尋)是非常重要的，而Bloom Filter是個在大量資料可以快速搜尋又省空間的方法，相較於其他的演算法他的搜尋只需要O(k)，k取決於用多少hash function，而且他佔用的空間也非常少，因此被利用在許多應用上，從惡意名單的找尋等等。簡單理解的話，如下圖，一開始會有兩個參數，分別整個需要查找的元素個數n及可以接受的false positive(誤報)，Bloom Filter會根據n,p，計算出總共需要的storage空間m及需要用到的hash function數目k，可以經由下面的鏈結簡單計算。

[

Bloom filter calculator
-----------------------

### Calculate the optimal size for your bloom filter, see how many items a given filter can hold, or just admire the curvy…

hur.st

](https://hur.st/bloomfilter/?source=post_page-----4493ba5425e9--------------------------------)

以下面為例，有一個數列{x,y,z}，假設使用到m的bit及k的hash func

新增
--

新增元素x，把元素做 hash(x)，做k次，把結果設為1，因此bitarray裡第1,5,13個bit被設為1。

搜尋
--

搜尋元素w，把元素做 hash(w)，做k次，檢查第4,13,15對應的bit是否全為1，任一bit為0此w不再集合中。

![](https://i.imgur.com/rmLpJnr.png)

程式實作
====

Bloom Filter
------------

首先，我使用的hash functiong是MurmurHash(mm3 library)，因為這個hash fuction可以根據seed來改變input的輸出值，一開始初始化BloomFilter的class: \_\_init\_\_ 會根據輸入的n, p計算出需要的空間m及hash func數目k。add及find可以參考上面的說明，可以看得出來Bloom Filter非常好實作。

> 請注意不要拿MurmurHash用做加密用途，加密請使用如SHA3等算法

智能合約 — 使用 solidity
------------------

由於避免在鏈上計算產生太多的費用，我們在鏈下計算再把Bloom Filter的結果上傳。這邊使用solidity撰寫，首先我們有幾個變數bf\_result, unised\_padding及manager，分別為Bloom Filter的bit array、轉成byte後多了多少的0及合約的管理者，一開始deploy合約會呼叫construstor()，把管理者設為deploy的人，若要存取的時候透過store的function來儲存Bloom Filter的bit array及padding zero，使用retrieve來讀取資料。

鏈上互動-web3.py
------------

由於web3.py目前無法直接跟區塊鏈錢包(Matamask等)互動，因此需要兩個東西，可以與鏈上互動的API及私鑰，為了簡化，我直接新增一個.env之後透過os.getenv(‘KEY’)，這樣可以直接透過w3.eth.account.privateKeyToAccount({KEY})直接呼叫，PROVIDER的部分可以把http://rinkeby.infura…這邊換成你自己本地運行的節點可以參考\[5–6\]更安全。

> 建議使用新建的測試錢包，直接把私鑰儲用明碼存在本機很危險。

**deploy 部署合約**

首先，我們需要將合約透過compile\_sol去做編譯，他會產生合約的ABI(Contract Application Binary Interface)及合約的二進位檔。

再來我們需要透過讀去合約ABI(contract\_interface\[‘abi’\])及二進位檔來建立一個合約My\_Contract。這邊藉由呼叫constructor()來建立一個transaction交易，並使用我們的錢包簽署(sign\_transactopn())最後送出可以參考下面code：

**store 儲存Bloom Filter**

首先我們需要初始化Bloom Filter(n, p)，把我們的檢查list新增成DataFrame格式data，依序新增到Bloom Filter，並透過合約地址及ABI初始化合約，並呼叫合約的store function，可以參考智能合約那節。跟deploy很像要先建立一筆交易、簽署，並使用send\_raw\_transcaction()送出交易。

**retrieve — 檢查資料是否在Bloom Filte**

最後我們直接透過呼叫合約(使用前面的provider)，呼叫智能合約的retrieve function，取得之前儲存Bloom Filter的bit array，b\_arr是在把bytes(byte array)還原成bit。

**Web App Prototype**

為了快速實踐構想，我使用了streamlit這個非常好用的套件，我這邊直接放上code及成品。

![](https://i.imgur.com/BERyZY7.png)


streamlit prototype

心得
==

這次期中我大概簡單實作了Bloom Filter及如何透過solidity及web3.py寫智能合約，很開心可以剛好利用這次的DApp期中把最近讀的paper的一小部分實作出來，蠻有成就感，因為這個可以解決在區塊鏈儲存的fee及快速的取得鏈上資訊，這個東西甚至可以被用在白/黑名單的確認上，期末可能在如何能讓別人的更便利使用上。

**以下是在上課進行的簡報(3頁，2min)**

Reference
=========

1.  [https://en.wikipedia.org/wiki/Bloom\_filter](https://en.wikipedia.org/wiki/Bloom_filter)
2.  [https://www.geeksforgeeks.org/bloom-filters-introduction-and-python-implementation/](https://www.geeksforgeeks.org/bloom-filters-introduction-and-python-implementation/)
3.  [https://en.wikipedia.org/wiki/MurmurHash](https://en.wikipedia.org/wiki/MurmurHash)
4.  [https://web3py.readthedocs.io/en/stable/contracts.html](https://web3py.readthedocs.io/en/stable/contracts.html)
5.  [https://web3py.readthedocs.io/en/stable/providers.html](https://web3py.readthedocs.io/en/stable/providers.html)
6.  [https://geth.ethereum.org/downloads/](https://geth.ethereum.org/downloads/)
7.  [https://web3py.readthedocs.io/en/stable/web3.eth.account.html?highlight=sign#sign-a-contract-transaction](https://web3py.readthedocs.io/en/stable/web3.eth.account.html?highlight=sign#sign-a-contract-transaction)

## Ref
- https://medium.com/@alanhc/%E4%BD%BF%E7%94%A8bloom-filter%E5%8F%8Aweb3-py%E4%BE%86%E5%81%9A%E7%96%AB%E8%8B%97%E8%AA%8D%E8%AD%89-ntu-dapp-%E6%9C%9F%E4%B8%ADprototype-4493ba5425e9

