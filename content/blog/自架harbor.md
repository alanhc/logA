---
title: 自架harbor
date: 2024-10-13
tags:
  - docker
updated: 2024-10-1322:18
up:
---



wget https://github.com/goharbor/harbor/releases/download/v2.11.1/harbor-online-installer-v2.11.1.tgz 

tar xzvf harbor-online-installer-v2.11.1.tgz 
cd harbor
Config HTTPS

使用OpenSSL 建立ＣＡ
openssl genrsa -out ca.key 4096

openssl req -x509 -new -nodes -sha512 -days 3650 \
 -subj "/C=TW/ST=Taiwan/L=Taipei/O=alanhc/OU=Personal/CN=0xfanslab.com" \
 -key ca.key \
 -out ca.crt

openssl genrsa -out 0xfanslab.com.key 4096

openssl req -sha512 -new \
    -subj "/C=TW/ST=Taiwan/L=Taipei/O=alanhc/OU=Personal/CN=0xfanslab.com" \
    -key 0xfanslab.com.key \
    -out 0xfanslab.com.csr

cat > v3.ext <<-EOF
authorityKeyIdentifier=keyid,issuer
basicConstraints=CA:FALSE
keyUsage = digitalSignature, nonRepudiation, keyEncipherment, dataEncipherment
extendedKeyUsage = serverAuth
subjectAltName = @alt_names

[alt_names]
DNS.1=0xfanslab.com
DNS.2=www.0xfanslab.com
DNS.3=localhost
EOF

authorityKeyIdentifier=keyid,issuer
basicConstraints=CA:FALSE
keyUsage = digitalSignature, nonRepudiation, keyEncipherment, dataEncipherment
extendedKeyUsage = serverAuth
subjectAltName = @alt_names

[alt_names]
DNS.1=0xfanslab.com
DNS.2=www.0xfanslab.com
DNS.3=localhost
IP.1=192.168.64.2 # multipass 要加上這


openssl x509 -req -sha512 -days 3650 \
    -extfile v3.ext \
    -CA ca.crt -CAkey ca.key -CAcreateserial \
    -in 0xfanslab.com.csr \
    -out 0xfanslab.com.crt



sudo mkdir -p /data/certs/      


sudo cp 0xfanslab.com.crt /data/certs/
sudo cp 0xfanslab.com.key /data/certs/

openssl x509 -inform PEM -in 0xfanslab.com.crt -out 0xfanslab.com.cert

sed -i 's|certificate: /your/certificate/path|certificate: '"$(pwd)/0xfanslab.com.crt"'|' harbor.yml
sed -i 's|private_key: /your/private/key/path|private_key: '"$(pwd)/0xfanslab.com.key"'|' harbor.yml
read -p "請輸入新的 Harbor 管理員密碼: " new_password
sed -i "s|harbor_admin_password: .*|harbor_admin_password: $new_password|" harbor.yml
sed -i "s|hostname: reg.mydomain.com|hostname: harbor.0xfanslab.com|" harbor.yml


sudo mkdir -p /etc/docker/certs.d/0xfanslab.com/
sudo cp 0xfanslab.com.cert /etc/docker/certs.d/0xfanslab.com/
sudo cp 0xfanslab.com.key /etc/docker/certs.d/0xfanslab.com/
sudo cp ca.crt /etc/docker/certs.d/0xfanslab.com/

sudo systemctl restart docker   


cp harbor.yml.tmpl harbor.yml  


設定
# https related config
https:
  # https port for harbor, default is 443
  port: 443
  # The path of cert and key files for nginx
  certificate: /Users/alantseng/harbor/0xfanslab.com.crt
  private_key: /Users/alantseng/harbor/0xfanslab.com.key

 harbor_admin_password:

./prepare

sudo docker compose down

 如果有錯：WARN[0000] /home/ubuntu/harbor/docker-compose.yml: the attribute `version` is obsolete, it will be ignored, please remove it to avoid potential confusion 
sed -i '/^version:/d' /home/ubuntu/harbor/docker-compose.yml

sudo docker compose up -d


Cloudflare 連線不到是要勾 No TLS Verify