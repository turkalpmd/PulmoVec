#!/bin/bash

# HeAR HuggingFace Authentication Setup Script

cat << 'EOF'
╔══════════════════════════════════════════════════════════════════════════╗
║           🔐 HeAR HuggingFace Authentication Setup                      ║
╚══════════════════════════════════════════════════════════════════════════╝
EOF

echo ""
echo "This script will help you authenticate with HuggingFace for the HeAR model."
echo ""

# Check if huggingface-cli is installed
if ! command -v huggingface-cli &> /dev/null; then
    echo "⚠️  huggingface-cli not found. Installing..."
    pip install huggingface-hub --upgrade
    echo ""
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 1: Request Access to HeAR Model"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Before authentication, you MUST request access to the HeAR model:"
echo ""
echo "  🌐 Visit: https://huggingface.co/google/hear-pytorch"
echo ""
echo "  👉 Click the 'Request Access' button"
echo "  ⏳ Wait for approval (usually instant or within a few minutes)"
echo ""
read -p "Have you requested and received access? (y/n): " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "❌ Please request access first, then run this script again."
    echo ""
    echo "Visit: https://huggingface.co/google/hear-pytorch"
    echo ""
    exit 1
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 2: Get Your HuggingFace Token"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "If you don't have a token yet:"
echo ""
echo "  🌐 Visit: https://huggingface.co/settings/tokens"
echo "  ➕ Click 'New token'"
echo "  📝 Give it a name (e.g., 'hear-training')"
echo "  ✅ Select 'read' permissions"
echo "  💾 Click 'Generate a token'"
echo "  📋 Copy the token (starts with 'hf_...')"
echo ""
read -p "Press Enter when you have your token ready..."
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 3: Login with HuggingFace CLI"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "You will now be prompted to enter your token."
echo "Paste your token when asked (it won't be visible as you type)."
echo ""

# Run huggingface-cli login
huggingface-cli login

if [ $? -eq 0 ]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "✅ Authentication Successful!"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "You're now ready to train the HeAR model!"
    echo ""
    echo "Next step:"
    echo "  $ python train_hear_classifier.py"
    echo ""
else
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "❌ Authentication Failed"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "Please check:"
    echo "  1. Your token is correct (starts with 'hf_...')"
    echo "  2. You have requested access to google/hear-pytorch"
    echo "  3. Your access request has been approved"
    echo ""
    echo "Try again: ./setup_hf_auth.sh"
    echo ""
    exit 1
fi
