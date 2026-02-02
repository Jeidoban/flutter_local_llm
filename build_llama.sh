#!/bin/bash

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}Starting llama.cpp build process...${NC}"

# Change to the llama.cpp directory
cd src/llama.cpp

# Run the build script and wait for it to finish
echo -e "${BLUE}Building llama.cpp xcframework...${NC}"
./build-xcframework.sh

# Check if the build was successful
if [ ! -d "build-apple/llama.xcframework" ]; then
    echo -e "${RED}Error: build-apple/llama.xcframework not found. Build may have failed.${NC}"
    exit 1
fi

echo -e "${GREEN}Build completed successfully!${NC}"

# Go back to the root directory
cd ../..

# Create a temporary directory for extraction
TEMP_DIR=$(mktemp -d)
echo -e "${BLUE}Extracting macOS framework from xcframework...${NC}"

# Find the macOS framework inside the xcframework
MACOS_FRAMEWORK=$(find src/llama.cpp/build-apple/llama.xcframework -name "llama.framework" -path "*macos*" | head -n 1)

if [ -z "$MACOS_FRAMEWORK" ]; then
    echo -e "${RED}Error: Could not find macOS framework in xcframework${NC}"
    rm -rf "$TEMP_DIR"
    exit 1
fi

# Copy the macOS framework to the plugin's macOS Frameworks directory
echo -e "${BLUE}Copying macOS framework to macos/Frameworks/...${NC}"
rm -rf macos/Frameworks/llama.framework
cp -R "$MACOS_FRAMEWORK" macos/Frameworks/

# Copy the xcframework to the iOS Frameworks directory
echo -e "${BLUE}Copying xcframework to ios/Frameworks/...${NC}"
rm -rf ios/Frameworks/llama.xcframework
cp -R src/llama.cpp/build-apple/llama.xcframework ios/Frameworks/

# Remove macOS framework from the iOS xcframework
echo -e "${BLUE}Removing macOS framework from iOS xcframework...${NC}"
find ios/Frameworks/llama.xcframework -type d -name "*macos*" -exec rm -rf {} + 2>/dev/null || true

# Update the Info.plist to remove macOS entry
echo -e "${BLUE}Updating xcframework Info.plist to remove macOS entry...${NC}"
PLIST_PATH="ios/Frameworks/llama.xcframework/Info.plist"

if [ -f "$PLIST_PATH" ]; then
    # Find and remove the macOS library entry from AvailableLibraries array
    # We'll do this by recreating the plist with only iOS entries
    LIBRARY_COUNT=$(/usr/libexec/PlistBuddy -c "Print :AvailableLibraries" "$PLIST_PATH" | grep -c "Dict" || echo "0")

    # Remove entries in reverse order to avoid index shifting
    for ((i=$LIBRARY_COUNT-1; i>=0; i--)); do
        PLATFORM=$(/usr/libexec/PlistBuddy -c "Print :AvailableLibraries:$i:SupportedPlatform" "$PLIST_PATH" 2>/dev/null || echo "")
        if [ "$PLATFORM" = "macos" ]; then
            /usr/libexec/PlistBuddy -c "Delete :AvailableLibraries:$i" "$PLIST_PATH"
            echo -e "${GREEN}Removed macOS entry from Info.plist${NC}"
        fi
    done
fi

# Clean up
rm -rf "$TEMP_DIR"

echo -e "${GREEN}Done! Frameworks have been placed in:${NC}"
echo -e "  - macOS framework: ${GREEN}macos/Frameworks/llama.framework${NC}"
echo -e "  - iOS xcframework (iOS only): ${GREEN}ios/Frameworks/llama.xcframework${NC}"
