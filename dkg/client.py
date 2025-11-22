from dkg.constants import BlockchainIds

# Print all available blockchain IDs
print("Available Blockchain IDs:")
for blockchain_id in BlockchainIds:
    print(f"  - {blockchain_id.name}: {blockchain_id.value}")