# Omi Hackathon Client Walkthrough

## 1. Overview
This is a React Native (Expo) app for interacting with the Omi backend. You can send test webhooks and view AI-generated responses.

---

## 2. Setup
1. Install dependencies:
   ```sh
   npm install
   ```
2. Start the Expo app:
   ```sh
   npx expo start
   ```
3. Scan the QR code with Expo Go (Android/iOS) or run on an emulator/simulator.

---

## 3. Using on a Physical Device
- **Android:**
  - Yes, enable USB debugging if you want to use a direct device connection (ADB) or run a development build.
  - For Expo Go, you can scan the QR code without USB debugging, but both your phone and PC must be on the same Wi-Fi network.
- **iOS:**
  - Use Expo Go and scan the QR code. No USB debugging needed.

---

## 4. Webhook Demo Screen
- Navigate to `/webhook` in the app to access the webhook test UI.
- Enter a user ID, transcript, and summary, then send to see the backend response.

---

## 5. Troubleshooting
- If the app can't reach your backend, ensure your phone and PC are on the same network and use your PC's local IP address in the API URL.
- For Android development builds, USB debugging is required for ADB install and log access.

---

## 6. Resources
- [Expo Docs](https://docs.expo.dev/)
- [React Native Docs](https://reactnative.dev/)
