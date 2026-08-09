#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

echo "======================================================="
echo "   FedChain Infrastructure Setup (Ubuntu / WSL)        "
echo "======================================================="

# --- 1. System Update & Dependencies ---
echo "[1/4] Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y wget curl git tar jq build-essential

# --- 2. Install Foundry (Anvil for Local Blockchain) ---
echo "[2/4] Installing Foundry (forge, cast, anvil, chisel)..."
# Download and run the official Foundry installation script
curl -L https://getfoundry.sh/install | bash

# Source the bash profile so foundryup is available in the current script session
if [ -f "$HOME/.bashrc" ]; then
    source "$HOME/.bashrc"
fi

# Run foundryup to install the latest pre-compiled binaries
# Explicitly using the path in case 'source' doesn't immediately expose it
$HOME/.foundry/bin/foundryup

echo "Foundry installed successfully. Anvil version:"
$HOME/.foundry/bin/anvil --version

# --- 3. Install Kubo (IPFS Node) ---
echo "[3/4] Installing Kubo (IPFS)..."
# Fetch the latest Kubo release version dynamically from GitHub
KUBO_VERSION=$(curl -s https://api.github.com/repos/ipfs/kubo/releases/latest | jq -r .tag_name)
echo "Latest Kubo version is $KUBO_VERSION"

# Download the tarball
cd /tmp
wget -q "https://github.com/ipfs/kubo/releases/download/${KUBO_VERSION}/kubo_${KUBO_VERSION}_linux-amd64.tar.gz"

# Extract and install
tar -xvzf "kubo_${KUBO_VERSION}_linux-amd64.tar.gz"
cd kubo
sudo bash install.sh

# Initialize the IPFS repository for the current user
echo "Initializing IPFS repository..."
# The '|| true' prevents failure if it's already initialized from a previous run
ipfs init || true 

echo "Kubo (IPFS) installed successfully. Version:"
ipfs --version

# --- 4. Create the Infrastructure Startup Script ---
echo "[4/4] Creating 'start_nodes.sh' helper script..."
cd - > /dev/null # Go back to the original directory (project root)

cat << 'EOF' > start_nodes.sh
#!/bin/bash

echo "Starting FedChain Background Infrastructure..."

# 1. Start Anvil (Local Blockchain) in the background
echo "-> Starting Anvil (RPC: http://127.0.0.1:8545)..."
# We pipe output to a log file so it doesn't clutter your terminal
nohup anvil > anvil_node.log 2>&1 &
ANVIL_PID=$!

# 2. Start IPFS Daemon in the background
echo "-> Starting IPFS Daemon (API: http://127.0.0.1:5001)..."
# Run the daemon offline (optional, but good for local dev without peering issues)
nohup ipfs daemon --offline > ipfs_node.log 2>&1 &
IPFS_PID=$!

echo ""
echo "✅ Infrastructure is running!"
echo "Anvil PID: $ANVIL_PID"
echo "IPFS PID:  $IPFS_PID"
echo ""
echo "You can now run: python main.py --config configs/exp4_fedchain.yaml"
echo ""
echo "To stop these services later, run:"
echo "kill $ANVIL_PID $IPFS_PID"
EOF

chmod +x start_nodes.sh

echo "======================================================="
echo "✅ Setup Complete!"
echo "======================================================="
echo "Please run: source ~/.bashrc (or restart your terminal) to update your PATH."
echo "Before running your Python experiments, start the nodes by running:"
echo "./start_nodes.sh"