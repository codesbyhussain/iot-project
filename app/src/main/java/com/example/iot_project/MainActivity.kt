package com.example.iot_project

import android.os.Bundle
import android.view.View
import android.widget.Button
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity

class MainActivity : AppCompatActivity() {

    private lateinit var passwordDisplay: TextView
    private lateinit var predictionDisplay: TextView
    private var currentPassword = StringBuilder()

    private val commonPasswords = listOf("1234", "0000", "1111", "1212", "7777", "1004", "2000", "4444", "123456")

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        passwordDisplay = findViewById(R.id.password_display)
        predictionDisplay = findViewById(R.id.prediction_display)
    }

    fun onNumberClick(view: View) {
        if (view is Button) {
            val number = view.text.toString()
            currentPassword.append(number)
            updateDisplay()
        }
    }

    fun onClearClick(view: View) {
        currentPassword.clear()
        updateDisplay()
    }

    fun onBackspaceClick(view: View) {
        if (currentPassword.isNotEmpty()) {
            currentPassword.deleteCharAt(currentPassword.length - 1)
            updateDisplay()
        }
    }

    private fun updateDisplay() {
        passwordDisplay.text = if (currentPassword.isEmpty()) "Enter password" else currentPassword.toString()
        predictPassword()
    }

    private fun predictPassword() {
        val predictions = commonPasswords.filter { it.startsWith(currentPassword.toString()) }
        if (predictions.isNotEmpty() && currentPassword.isNotEmpty()) {
            predictionDisplay.text = "Predictions: " + predictions.joinToString(", ")
        } else {
            predictionDisplay.text = "No predictions"
        }
    }
}
