from dkg import DKG
from dkg.providers import BlockchainProvider, NodeHTTPProvider
from dkg.constants import BlockchainIds
import os

private_key = os.getenv('PRIVATE_KEY')
if not private_key:
    print("❌ PRIVATE_KEY not found in environment variables")
    print("Please set it with: export PRIVATE_KEY=your_private_key_here")

blockchain_provider = BlockchainProvider(BlockchainIds.NEUROWEB_TESTNET.value)
node_provider = NodeHTTPProvider(endpoint_uri="http://localhost:8900", api_version="v1")
# Initialize DKG with only blockchain_provider for edge mode
dkg = DKG(blockchain_provider=blockchain_provider, node_provider=node_provider)

print("✅ DKG Edge Node initialized successfully")