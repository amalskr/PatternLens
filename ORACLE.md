# Free 24/7 hosting on Oracle Cloud Always Free

One small Linux server runs everything: the page, the API, the forward log and the Telegram alerts.
Always Free resources cost nothing. Oracle asks for a card to verify you; it is not charged unless you
upgrade the account and create paid resources.

## 1. Account (once)

1. Sign up at oracle.com/cloud/free.
2. **Home region:** choose carefully, it cannot be changed and Always Free servers only run there.
   Closest to Sri Lanka: India South (Hyderabad), India West (Mumbai) or Singapore.
   Binance blocks some countries; the test in step 4 tells you if the server can reach it.

## 2. SSH key (on your Mac, once)

```bash
ssh-keygen -t ed25519
cat ~/.ssh/id_ed25519.pub
```

Copy the printed line; you paste it in the next step.

## 3. Create the server

Oracle console > Compute > Instances > Create instance:

- Image: **Ubuntu 24.04**
- Shape: **VM.Standard.A1.Flex** (Ampere, "Always Free eligible"), 1 OCPU and 6 GB memory
  (if it says "out of capacity", try again later or another availability domain)
- SSH keys: paste the key from step 2
- Create, then copy the **public IP address**

Open port 8000 so your browser can reach it: on the instance page open the subnet > Security List >
Add Ingress Rule: source `0.0.0.0/0`, TCP, destination port `8000`.

## 4. Prepare the server

```bash
ssh ubuntu@PUBLIC_IP
```

On the server:

```bash
sudo apt update && sudo apt install -y python3-venv python3-pip libgomp1 iptables-persistent
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 8000 -j ACCEPT
sudo netfilter-persistent save
curl -s https://fapi.binance.com/fapi/v1/time
exit
```

The last command must print `{"serverTime":...}`. If it prints an error or a "restricted location"
message, Binance blocks this region: the page still works (your browser fetches the chart itself), but
keep the forward log and alerts running on your Mac.

## 5. Copy the project from your Mac

Stop the forward log loop on your Mac first (Ctrl + C), so the log is not written in two places. Then:

```bash
cd ~/PycharmProjects/PatternLens
rsync -av --exclude .venv --exclude data --exclude .git --exclude __pycache__ --exclude .idea \
  ./ ubuntu@PUBLIC_IP:~/PatternLens/
```

This copies the models, the research results and your forward log so far.

## 6. Install and start (on the server)

```bash
ssh ubuntu@PUBLIC_IP
cd PatternLens
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
sed 's/^export //' alerts.env > pl.env
echo "PAGE_PASSWORD=choose_a_password" >> pl.env
sudo cp deploy/pattern-lens-api.service deploy/pattern-lens-log.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now pattern-lens-api pattern-lens-log
```

Both start again by themselves after a crash or a reboot.

Open `http://PUBLIC_IP:8000` on your phone or Mac. The browser asks for a user name (anything) and the
password from `pl.env`.

## Everyday commands (on the server)

```bash
sudo systemctl status pattern-lens-api pattern-lens-log
journalctl -u pattern-lens-log -f
.venv/bin/python forward_log.py --report
```

## Updating after a change on your Mac

```bash
rsync -av --exclude .venv --exclude data --exclude .git --exclude __pycache__ --exclude .idea \
  --exclude results/forward_log.sqlite --exclude results/alert_state.json --exclude pl.env \
  ./ ubuntu@PUBLIC_IP:~/PatternLens/
ssh ubuntu@PUBLIC_IP "sudo systemctl restart pattern-lens-api pattern-lens-log"
```

The excludes keep the server's live forward log and settings.

## Backup the forward log to your Mac (now and then)

```bash
scp ubuntu@PUBLIC_IP:~/PatternLens/results/forward_log.sqlite results/forward_log_backup.sqlite
```

## Good to know

- The page is plain HTTP. The password is enough to keep strangers out, but it is not encrypted.
  For HTTPS without a domain: `cloudflared tunnel --url http://localhost:8000` on the server gives a
  temporary https address.
- Oracle may stop Always Free servers it considers idle for a week. You get an email and can start it
  again; the data stays. The backup above covers you either way.
