import { useState, useEffect } from "react";
import { SafeAreaView } from "react-native-safe-area-context";
import { TouchableOpacity, View, Text, ScrollView, Alert, Platform } from "react-native";
import {
  useSpeechRecognitionEvent,
  ExpoSpeechRecognitionModule,
  AudioEncodingAndroid,
} from "expo-speech-recognition";

const HomeScreen = () => {
  const [isRecognizing, setIsRecognizing] = useState(false);
  const [transcript, setTranscript] = useState<string[]>([]);
  const [currentTranscript, setCurrentTranscript] = useState("");
  const [elapsedTime, setElapsedTime] = useState(0);

  // Timer effect
  useEffect(() => {
    let interval: ReturnType<typeof setInterval>;
    if (isRecognizing) {
      interval = setInterval(() => {
        setElapsedTime((prev) => prev + 1);
      }, 1000);
    } else {
      setElapsedTime(0);
    }
    return () => clearInterval(interval);
  }, [isRecognizing]);

  // Listen for speech recognition results
  useSpeechRecognitionEvent("result", (event) => {
    const transcribedText = event.results[0]?.transcript;
    if (transcribedText) {
      console.log("📝 Transcribed:", transcribedText);
      setCurrentTranscript(transcribedText);
    }
  });

  // Listen for final results
  useSpeechRecognitionEvent("end", () => {
    console.log("🛑 Recognition ended");
    if (currentTranscript) {
      const timestamp = new Date().toLocaleTimeString();
      const logEntry = `[${timestamp}] ${currentTranscript}`;
      console.log("✅ Final transcript:", logEntry);
      setTranscript((prev) => [...prev, logEntry]);
      setCurrentTranscript("");
    }
    setIsRecognizing(false);
  });

  // Listen for errors
  useSpeechRecognitionEvent("error", (event) => {
    console.error("❌ Speech recognition error:", event.error);
    Alert.alert("Error", `Speech recognition failed: ${event.error}`);
    setIsRecognizing(false);
  });

  const handleStartRecognition = async () => {
    try {
      console.log("🎙️ Starting speech recognition...");

      // Request permissions
      const { status } = await ExpoSpeechRecognitionModule.requestPermissionsAsync();
      if (status !== "granted") {
        Alert.alert(
          "Permission Required",
          "Microphone permission is required for speech recognition"
        );
        return;
      }

      // Start speech recognition
      await ExpoSpeechRecognitionModule.start({
        lang: "en-US",
        interimResults: true,
        maxAlternatives: 1,
        continuous: true,
        requiresOnDeviceRecognition: false,
        addsPunctuation: true,
        contextualStrings: ["news", "reporter", "interview"],
      });

      setIsRecognizing(true);
      const startLog = `[${new Date().toLocaleTimeString()}] 🎙️ Recording started - speak now`;
      setTranscript((prev) => [...prev, startLog]);
      console.log("✅ Speech recognition started");
    } catch (error) {
      console.error("❌ Error starting recognition:", error);
      Alert.alert("Error", "Failed to start speech recognition: " + error);
      setIsRecognizing(false);
    }
  };

  const handleStopRecognition = async () => {
    try {
      console.log("⏹️ Stopping speech recognition...");
      await ExpoSpeechRecognitionModule.stop();
      
      const stopLog = `[${new Date().toLocaleTimeString()}] ⏹️ Recording stopped`;
      setTranscript((prev) => [...prev, stopLog]);
      console.log("✅ Speech recognition stopped");
    } catch (error) {
      console.error("❌ Error stopping recognition:", error);
      Alert.alert("Error", "Failed to stop speech recognition: " + error);
    }
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
  };

  return (
    <SafeAreaView className="bg-slate-950 flex-1">
      <ScrollView className="flex-1" contentContainerStyle={{ paddingBottom: 20 }}>
        <View className="flex-1 justify-center items-center px-6 pt-8">
          {/* App Title */}
          <Text className="text-white text-3xl font-bold mb-2">
            News Recorder
          </Text>
          <Text className="text-slate-400 text-base mb-8">
            Real-time voice transcription
          </Text>

          {/* Recording Status Indicator */}
          <View className="mb-8 items-center">
            {isRecognizing ? (
              <View className="items-center">
                <View className="bg-red-500 w-4 h-4 rounded-full mb-3" />
                <Text className="text-red-500 text-lg font-semibold">
                  Recording...
                </Text>
                <Text className="text-white text-4xl font-mono mt-4">
                  {formatTime(elapsedTime)}
                </Text>
              </View>
            ) : (
              <View className="items-center">
                <View className="bg-slate-700 w-4 h-4 rounded-full mb-3" />
                <Text className="text-slate-400 text-lg">
                  Ready to Record
                </Text>
              </View>
            )}
          </View>

          {/* Live Transcript Preview */}
          {currentTranscript && (
            <View className="w-full mb-6 bg-slate-800 rounded-xl p-4">
              <Text className="text-slate-400 text-sm mb-2">Live:</Text>
              <Text className="text-white text-base">{currentTranscript}</Text>
            </View>
          )}

          {/* Recording Controls */}
          <View className="w-full mt-8">
            {!isRecognizing ? (
              <TouchableOpacity
                onPress={handleStartRecognition}
                className="bg-purple-600 rounded-2xl py-5 px-8 shadow-2xl shadow-purple-600/50 active:bg-purple-700"
              >
                <Text className="text-white text-center text-xl font-bold">
                  Start Recording
                </Text>
              </TouchableOpacity>
            ) : (
              <TouchableOpacity
                onPress={handleStopRecognition}
                className="bg-slate-800 border-2 border-red-500 rounded-2xl py-5 px-8 active:bg-slate-700"
              >
                <Text className="text-red-500 text-center text-xl font-bold">
                  Stop Recording
                </Text>
              </TouchableOpacity>
            )}
          </View>

          {/* Info Text */}
          <View className="mt-12 items-center">
            <Text className="text-slate-500 text-sm text-center px-8">
              {isRecognizing
                ? "Speak now. Your words will be transcribed in real-time."
                : "Tap Start to begin recording and transcribing your speech"}
            </Text>
          </View>

          {/* Transcript Log Section */}
          {transcript.length > 0 && (
            <View className="w-full mt-12 bg-slate-900 rounded-xl p-4">
              <View className="flex-row items-center justify-between mb-3">
                <Text className="text-white text-lg font-semibold">
                  Transcript Log
                </Text>
                <TouchableOpacity onPress={() => setTranscript([])}>
                  <Text className="text-purple-500 text-sm">Clear</Text>
                </TouchableOpacity>
              </View>
              <ScrollView className="max-h-80">
                {transcript.map((entry, index) => (
                  <Text
                    key={index}
                    className="text-sm text-slate-300 mb-2 leading-6"
                  >
                    {entry}
                  </Text>
                ))}
              </ScrollView>
            </View>
          )}
        </View>
      </ScrollView>
    </SafeAreaView>
  );
};

export default HomeScreen;
