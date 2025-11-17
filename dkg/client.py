from dkg import DKG
from dkg.providers import BlockchainProvider, NodeHTTPProvider

node_provider = NodeHTTPProvider(endpoint_uri="http://localhost:8900", api_version="v1")
blockchain_provider = BlockchainProvider(
    Environments.DEVELOPMENT.value, # or TESTNET, MAINNET
    BlockchainIds.HARDHAT_1.value,
)

dkg = DKG(node_provider, blockchain_provider)

print(dkg.node.info)
# if successfully connected, this should print the dictionary with node version
# { "version": "8.X.X" }