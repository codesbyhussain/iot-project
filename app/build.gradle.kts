plugins {
    id("com.android.application")
    kotlin("android")
}

android {
    namespace = "com.example.sensorclassifier"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.example.sensorclassifier"
        minSdk = 26
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"

        // Only include these ABIs to avoid incompatible .so files
        ndk {
            abiFilters += listOf("armeabi-v7a", "arm64-v8a")
        }
    }

    buildTypes {
        getByName("release") {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }

    // Prevent compression of .tflite files
    aaptOptions {
        noCompress("tflite")
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }
}

dependencies {
    // TensorFlow Lite CPU only (16 KB page-aligned libraries)
    implementation("org.tensorflow:tensorflow-lite:2.15.0")
    // AndroidX Core & AppCompat
    implementation("androidx.core:core-ktx:1.10.1")
    implementation("androidx.appcompat:appcompat:1.6.1")

    // Material Components
    implementation("com.google.android.material:material:1.9.0")
}
