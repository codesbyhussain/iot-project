// Top-level build file where you can add configuration options common to all sub-projects/modules.
plugins {
    alias(libs.plugins.android.application) apply false
    alias(libs.plugins.kotlin.android) apply false
}
// Project-level build.gradle.kts (minimal, Kotlin DSL)

buildscript {
    repositories {
        google()
        mavenCentral()
    }
    dependencies {
        // Android Gradle Plugin
        classpath("com.android.tools.build:gradle:8.13.1")
        // Kotlin Gradle Plugin (if you use Kotlin)
        classpath("org.jetbrains.kotlin:kotlin-gradle-plugin:1.9.10")
    }
}

tasks.register("clean", Delete::class) {
    delete(rootProject.buildDir)
}
