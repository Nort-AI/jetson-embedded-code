#!/bin/bash
# Install nort as a systemd service
set -e
INSTALL_DIR=/opt/nort

echo "Installing Nort Analytics Edge Agent..."
sudo cp system/nort.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable nort
sudo systemctl start nort
echo "Done. Check status with: systemctl status nort"
