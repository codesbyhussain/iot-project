package com.example.sensorclassifier

import android.app.Activity
import android.os.Bundle
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.content.Context
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

class MainActivity : Activity(), SensorEventListener {

    private lateinit var sensorManager: SensorManager

    // Sensors for each digit
    private val accel = FloatArray(3)
    private val gyro = FloatArray(3)
    private val rotation = FloatArray(3)
    private val magnetic = FloatArray(3)

    private val accel2 = FloatArray(3)
    private val gyro2 = FloatArray(3)
    private val rotation2 = FloatArray(3)
    private val magnetic2 = FloatArray(3)

    private val accel3 = FloatArray(3)
    private val gyro3 = FloatArray(3)
    private val rotation3 = FloatArray(3)
    private val magnetic3 = FloatArray(3)

    private val accel4 = FloatArray(3)
    private val gyro4 = FloatArray(3)
    private val rotation4 = FloatArray(3)
    private val magnetic4 = FloatArray(3)

    private var enteredDigits = mutableListOf<String>()

    // ----------------- Models -----------------
    private var modelCurrentDigit: Interpreter? = null
    private var model2Digit: Interpreter? = null
    private var model3Digit: Interpreter? = null
    private var model4Digit: Interpreter? = null

    private var model3SeqStage1: Interpreter? = null
    private var model3SeqStage2: Interpreter? = null
    private var model3SeqStage3: Interpreter? = null

    private var model4SeqStage1: Interpreter? = null
    private var model4SeqStage2: Interpreter? = null
    private var model4SeqStage3: Interpreter? = null
    private var model4SeqStage4: Interpreter? = null

    // ----------------- UI Elements -----------------
    private lateinit var tvEntered: TextView
    private lateinit var tvCurrentDigit: TextView
    private lateinit var tvNextDigit: TextView
    private lateinit var tv3Seq: TextView
    private lateinit var tv4Seq: TextView

    // ----------------- Sequence Lookup Tables -----------------
    // NOTE: This list MUST EXACTLY MATCH the order of the pins used when fitting the LabelEncoder in Python!
    private val SEQUENCE_LOOKUP_4DIGIT_25 = listOf(
        "2148","7255","7932","5820","7513",
        "4091","0367","1285","6702","3849",
        "5974","8321","0456","2197","6803",
        "1509","6472","3084","9720","5316",
        "8602","4938","1750","2461","9375"
    )

    private val SEQUENCE_LOOKUP_3DIGIT_25 = listOf(
        "701","327","540","913","647",
        "480","159","230","086","472",
        "395","028","614","207","851",
        "934","120","576","482","309",
        "068","743","219","805","436",
        "999", "998", "997", "996", "995",
        "994", "993", "992", "991", "990",
        "899", "898", "897", "896", "895",
        "894", "893", "892", "891", "890",
        "799", "798", "797", "796", "795"
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        registerSensor(Sensor.TYPE_ACCELEROMETER)
        registerSensor(Sensor.TYPE_GYROSCOPE)
        registerSensor(Sensor.TYPE_ROTATION_VECTOR)
        registerSensor(Sensor.TYPE_MAGNETIC_FIELD)

        // ----------------- UI -----------------
        tvEntered = findViewById(R.id.tvEntered)
        tvCurrentDigit = findViewById(R.id.tvCurrentDigit)
        tvNextDigit = findViewById(R.id.tvNextDigit)
        tv3Seq = findViewById(R.id.tv3Seq)
        tv4Seq = findViewById(R.id.tv4Seq)

        val btnIds = listOf(
            R.id.btn0,R.id.btn1,R.id.btn2,R.id.btn3,R.id.btn4,
            R.id.btn5,R.id.btn6,R.id.btn7,R.id.btn8,R.id.btn9
        )

        // ----------------- Digit Buttons -----------------
        for (id in btnIds) {
            findViewById<Button>(id).setOnClickListener { b ->
                val digit = (b as Button).text.toString()
                if (enteredDigits.size < 4) {
                    enteredDigits.add(digit)
                    tvEntered.text = enteredDigits.joinToString("")
                    runPredictions()
                }
            }
        }

        findViewById<Button>(R.id.btnClear).setOnClickListener {
            enteredDigits.clear()
            tvEntered.text = ""
            clearPredictions()
        }

        // ----------------- Load Models (using safe function) -----------------
        modelCurrentDigit = loadTFLiteModel("predict_digit_model.tflite")
        model2Digit = loadTFLiteModel("predict_2nd_digit_model.tflite")
        model3Digit = loadTFLiteModel("predict_3rd_digit_model.tflite")
        model4Digit = loadTFLiteModel("predict_4th_digit_model.tflite")

        model3SeqStage1 = loadTFLiteModel("tflite_3digit_seq_after_1.tflite")
        model3SeqStage2 = loadTFLiteModel("tflite_3digit_seq_after_2.tflite")
        model3SeqStage3 = loadTFLiteModel("tflite_3digit_seq_after_3.tflite")

        model4SeqStage1 = loadTFLiteModel("tflite_4digit_seq_after_1.tflite")
        model4SeqStage2 = loadTFLiteModel("tflite_4digit_seq_after_2.tflite")
        model4SeqStage3 = loadTFLiteModel("tflite_4digit_seq_after_3.tflite")
        model4SeqStage4 = loadTFLiteModel("tflite_4digit_seq_after_4.tflite")

        // Call debug function to print model shapes (run *after* loading)
        debugAllModels()
    }

    // ----------------- Sensor Registration -----------------
    private fun registerSensor(sensorType: Int) {
        sensorManager.getDefaultSensor(sensorType)?.also {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_NORMAL)
        }
    }

    // ---------------- Debug TFLite model shapes ----------------
    private fun debugAllModels() {
        val models = mapOf(
            modelCurrentDigit to "Current Digit",
            model2Digit to "2nd Digit",
            model3Digit to "3rd Digit",
            model4Digit to "4th Digit",
            model3SeqStage1 to "3-digit Seq Stage 1",
            model3SeqStage2 to "3-digit Seq Stage 2",
            model3SeqStage3 to "3-digit Seq Stage 3",
            model4SeqStage1 to "4-digit Seq Stage 1",
            model4SeqStage2 to "4-digit Seq Stage 2",
            model4SeqStage3 to "4-digit Seq Stage 3",
            model4SeqStage4 to "4-digit Seq Stage 4"
        )

        println("--- TFLite Model Debug: Expected Input/Output Shapes ---")
        models.forEach { (model, name) ->
            model?.let {
                println("===== $name =====")
                for (i in 0 until it.inputTensorCount) {
                    val t = it.getInputTensor(i)
                    println("Input $i shape: ${t.shape().contentToString()}, type: ${t.dataType()}")
                }
                for (i in 0 until it.outputTensorCount) {
                    val t = it.getOutputTensor(i)
                    println("Output $i shape: ${t.shape().contentToString()}, type: ${t.dataType()}")
                }
                println("===================")
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        // Close models gracefully
        modelCurrentDigit?.close()
        model2Digit?.close()
        model3Digit?.close()
        model4Digit?.close()
        model3SeqStage1?.close()
        model3SeqStage2?.close()
        model3SeqStage3?.close()
        model4SeqStage1?.close()
        model4SeqStage2?.close()
        model4SeqStage3?.close()
        model4SeqStage4?.close()
        sensorManager.unregisterListener(this)
    }

    private fun loadModelFile(filename: String): ByteBuffer {
        val afd = assets.openFd(filename)
        val inputStream = FileInputStream(afd.fileDescriptor)
        val channel = inputStream.channel
        return channel.map(FileChannel.MapMode.READ_ONLY, afd.startOffset, afd.declaredLength)
            .order(ByteOrder.nativeOrder())
    }

    // ----------------- HELPER FUNCTIONS -----------------

    private fun loadTFLiteModel(filename: String): Interpreter? {
        return try {
            val buffer = loadModelFile(filename)
            Interpreter(buffer)
        } catch (e: Exception) {
            val msg = "🚨 Error loading model: $filename. Check file name and existence in 'assets' folder. Cause: ${e.message}"
            println(msg)
            Toast.makeText(this, msg, Toast.LENGTH_LONG).show()
            null
        }
    }

    private fun getPinFromIndex(index: Int, sequenceType: String): String {
        // If index is -1 or -2, return the error code string
        if (index < 0) return "$index"

        // Output size 10 means it's a digit prediction
        if (sequenceType == "DIGIT") return index.toString()

        val lookup = when (sequenceType) {
            "3SEQ_25" -> SEQUENCE_LOOKUP_3DIGIT_25.subList(0, 25) // Only take the first 25
            "4SEQ_25" -> SEQUENCE_LOOKUP_4DIGIT_25
            else -> return "UNKNOWN SEQ TYPE"
        }

        // Lookup the pin string
        return if (index >= 0 && index < lookup.size) {
            lookup[index]
        } else {
            "UNKNOWN (Index $index)"
        }
    }

    private fun buildRawDigits(digits: List<String>): FloatArray {
        return digits.map { it.toFloat() }.toFloatArray()
    }

    private fun buildOneHotDigits(digits: List<String>): FloatArray {
        val array = FloatArray(digits.size * 10) { 0f }
        for (i in digits.indices) {
            val d = digits[i].toInt()
            array[10*i + d] = 1f
        }
        return array
    }

    private fun concat(vararg arrays: FloatArray): FloatArray {
        val total = arrays.sumOf { it.size }
        val result = FloatArray(total)
        var index = 0
        for (a in arrays) {
            for (v in a) {
                result[index++] = v
            }
        }
        return result
    }

    /**
     * Safely runs TFLite inference, accommodating dynamic output size.
     * @param outputSize The expected size of the output tensor (10 for digit, 25/50 for sequence).
     */
    private fun runTFLiteModel(model: Interpreter?, input: FloatArray, modelName: String, outputSize: Int): Int {
        if (model == null) return -1
        if (outputSize <= 0) return -3

        try {
            val inputBuffer = ByteBuffer.allocateDirect(input.size * 4).order(ByteOrder.nativeOrder())
            input.forEach { inputBuffer.putFloat(it) }
            inputBuffer.rewind()

            // DYNAMICALLY CREATE THE OUTPUT ARRAY
            val output = Array(1) { FloatArray(outputSize) }

            println("--- Attempting inference for: $modelName (Input Size: ${input.size}, Output Size: $outputSize) ---")

            model.run(inputBuffer, output)

            println("--- SUCCESSFUL inference for: $modelName ---")

            // Find the index of the maximum value (the predicted index/digit)
            return output[0].indices.maxByOrNull { output[0][it] } ?: -1

        } catch (e: Exception) {
            val msg = "💥 CRASH DETECTED: Model Inference Failed for $modelName. Check Logcat! Cause: ${e.message}"
            println(msg)
            Toast.makeText(this, msg, Toast.LENGTH_LONG).show()
            return -2
        }
    }

    // ----------------- Prediction Runner -----------------
    private fun runPredictions() {
        if (enteredDigits.isEmpty()) {
            clearPredictions()
            return
        }

        // Pre-calculate sensor data snapshots
        val sensorData0 = buildSensorInput(0)
        val sensorData1 = buildSensorInput(1)
        val sensorData2 = buildSensorInput(2)
        val sensorData3 = buildSensorInput(3)

        // Prepare Raw Digit Features (no longer needed for sequence models)
        // val rawDigits = buildRawDigits(enteredDigits)

        // Prepare One-Hot Digit Features (used by all multi-digit prediction models)
        val oheDigits = buildOneHotDigits(enteredDigits)


        // ----- Current Digit Prediction (Input Size: 12, Output Size: 10) -----
        val currentInput = sensorData0
        println("Input size for Current Digit Model: ${currentInput.size} (Expected: 12)")
        val currentPred = runTFLiteModel(modelCurrentDigit, currentInput, "Current Digit", 10)
        tvCurrentDigit.text = "Current Digit Pred: ${getPinFromIndex(currentPred, "DIGIT")}"

        // ----- Next Digit Prediction (RETAINED One-Hot LOGIC - Input Size: 22, 44, 66) -----
        val nextPred = when (enteredDigits.size) {
            1 -> {
                // Input Size: 10 OHE + 12 Sensors = 22
                val input = concat(oheDigits, sensorData0)
                println("Input size for Next Digit (2nd model) with 1 digit entered: ${input.size} (Expected: 10 + 12 = 22)")
                runTFLiteModel(model2Digit, input, "2nd Digit", 10)
            }
            2 -> {
                // Input Size: 20 OHE + 24 Sensors = 44
                val sensors = sensorData0 + sensorData1
                val input = concat(oheDigits, sensors)
                println("Input size for Next Digit (3rd model) with 2 digits entered: ${input.size} (Expected: 20 + 24 = 44)")
                runTFLiteModel(model3Digit, input, "3rd Digit", 10)
            }
            3 -> {
                // Input Size: 30 OHE + 36 Sensors = 66
                val sensors = sensorData0 + sensorData1 + sensorData2
                val input = concat(oheDigits, sensors)
                println("Input size for Next Digit (4th model) with 3 digits entered: ${input.size} (Expected: 30 + 36 = 66)")
                runTFLiteModel(model4Digit, input, "4th Digit", 10)
            }
            else -> -1
        }
        tvNextDigit.text = "Next Digit Pred: ${getPinFromIndex(nextPred, "DIGIT")}"

        // ----- 3-Digit Sequence Prediction (NOW USING ONE-HOT ENCODING) -----
        val pred3SeqIndex = when (enteredDigits.size) {
            1 -> {
                // Input Size: 10 OHE + 12 Sensors = 22
                val input = concat(oheDigits, sensorData0)
                println("Input size for 3-Digit Seq Stage 1: ${input.size} (Expected: 10 + 12 = 22)")
                runTFLiteModel(model3SeqStage1, input, "3-Digit Seq Stage 1", 25)
            }
            2 -> {
                // Input Size: 20 OHE + 24 Sensors = 44
                val sensors = sensorData0 + sensorData1
                val input = concat(oheDigits, sensors)
                println("Input size for 3-Digit Seq Stage 2: ${input.size} (Expected: 20 + 24 = 44)")
                runTFLiteModel(model3SeqStage2, input, "3-Digit Seq Stage 2", 25)
            }
            3 -> {
                // Input Size: 30 OHE + 36 Sensors = 66
                val sensors = sensorData0 + sensorData1 + sensorData2
                val input = concat(oheDigits, sensors)
                println("Input size for 3-Digit Seq Stage 3: ${input.size} (Expected: 30 + 36 = 66)")
                runTFLiteModel(model3SeqStage3, input, "3-Digit Seq Stage 3", 25)
            }
            else -> -1
        }
        // Translate the index into the actual pin
        tv3Seq.text = "3-Digit PIN Pred: ${getPinFromIndex(pred3SeqIndex, "3SEQ_25")}"

        // ----- 4-Digit Sequence Prediction (NOW USING ONE-HOT ENCODING) -----
        val pred4SeqIndex = when (enteredDigits.size) {
            1 -> {
                // Input Size: 10 OHE + 12 Sensors = 22
                val input = concat(oheDigits, sensorData0)
                println("Input size for 4-Digit Seq Stage 1: ${input.size} (Expected: 10 + 12 = 22)")
                runTFLiteModel(model4SeqStage1, input, "4-Digit Seq Stage 1", 25)
            }
            2 -> {
                // Input Size: 20 OHE + 24 Sensors = 44
                val sensors = sensorData0 + sensorData1
                val input = concat(oheDigits, sensors)
                println("Input size for 4-Digit Seq Stage 2: ${input.size} (Expected: 20 + 24 = 44)")
                runTFLiteModel(model4SeqStage2, input, "4-Digit Seq Stage 2", 25)
            }
            3 -> {
                // Input Size: 30 OHE + 36 Sensors = 66
                val sensors = sensorData0 + sensorData1 + sensorData2
                val input = concat(oheDigits, sensors)
                println("Input size for 4-Digit Seq Stage 3: ${input.size} (Expected: 30 + 36 = 66)")
                runTFLiteModel(model4SeqStage3, input, "4-Digit Seq Stage 3", 25)
            }
            4 -> {
                // Input Size: 40 OHE + 48 Sensors = 88
                val sensors = sensorData0 + sensorData1 + sensorData2 + sensorData3
                val input = concat(oheDigits, sensors)
                println("Input size for 4-Digit Seq Stage 4: ${input.size} (Expected: 40 + 48 = 88)")
                runTFLiteModel(model4SeqStage4, input, "4-Digit Seq Stage 4", 25)
            }
            else -> -1
        }
        // Translate the index into the actual pin
        tv4Seq.text = "4-Digit PIN Pred: ${getPinFromIndex(pred4SeqIndex, "4SEQ_25")}"
    }

    // ----------------- Input Builders (RETAINED) -----------------
    private fun buildSensorInput(digitIndex: Int): FloatArray {
        return when(digitIndex) {
            0 -> concat(accel, gyro, rotation, magnetic)
            1 -> concat(accel2, gyro2, rotation2, magnetic2)
            2 -> concat(accel3, gyro3, rotation3, magnetic3)
            3 -> concat(accel4, gyro4, rotation4, magnetic4)
            else -> FloatArray(12)
        }
    }

    private fun buildInputFor2ndDigit(): FloatArray {
        if (enteredDigits.isEmpty()) return FloatArray(22) { 0f } // safety

        val input = FloatArray(22)

        // One-hot encode first digit with weight 5
        val firstDigit = enteredDigits[0].toInt()
        input[firstDigit] = 5f

        // Append first digit's sensor readings
        val sensors = FloatArray(12)
        System.arraycopy(accel, 0, sensors, 0, 3)
        System.arraycopy(gyro, 0, sensors, 3, 3)
        System.arraycopy(rotation, 0, sensors, 6, 3)
        System.arraycopy(magnetic, 0, sensors, 9, 3)

        System.arraycopy(sensors, 0, input, 10, 12)

        return input
    }


    // ----------------- Sensor Callbacks (RETAINED) -----------------
    override fun onSensorChanged(event: SensorEvent) {
        when(event.sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> System.arraycopy(event.values, 0, accel, 0, 3)
            Sensor.TYPE_GYROSCOPE -> System.arraycopy(event.values, 0, gyro, 0, 3)
            Sensor.TYPE_ROTATION_VECTOR -> {
                rotation[0] = event.values.getOrElse(0){0f}
                rotation[1] = event.values.getOrElse(1){0f}
                rotation[2] = event.values.getOrElse(2){0f}
            }
            Sensor.TYPE_MAGNETIC_FIELD -> System.arraycopy(event.values, 0, magnetic, 0, 3)
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    private fun clearPredictions() {
        tvCurrentDigit.text = ""
        tvNextDigit.text = ""
        tv3Seq.text = ""
        tv4Seq.text = ""
    }
}