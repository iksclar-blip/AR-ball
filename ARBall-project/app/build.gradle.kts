plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.example.arball"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.example.arball"
        minSdk = 26
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
    androidResources { noCompress.add("task") }
}

dependencies {
    implementation("com.google.ar:core:1.44.0")
    implementation("com.google.mediapipe:tasks-vision:0.10.14")
}
