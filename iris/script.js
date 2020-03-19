import * as tfvis from '@tensorflow/tfjs-vis'
import * as tf from '@tensorflow/tfjs'
import { getIrisData, IRIS_CLASSES, IRIS_NUM_CLASSES } from './data.js'

window.onload = async () => {
    const [xTrain, yTrain, xTest, yTest] = getIrisData(0.15);

    const model = tf.sequential();

    // 构建模型
    model.add(tf.layers.dense({
        units: 10,
        inputShape: [xTrain.shape[1]],
        activation: "sigmoid"
    }))

    model.add(tf.layers.dense({
        units: 3,
        activation: "softmax"
    }))

    // 设置损失器和优化器
    model.compile({
        loss: 'categoricalCrossentropy',
        optimizer: tf.train.adam(.1),
        metrics: ['accuracy']
    })

    // 训练
    await model.fit(xTrain, yTrain, {
        epochs: 100,
        validationData: [xTest, yTest],
        callbacks: tfvis.show.fitCallbacks(
            { name: '多分类问题训练效果' },
            ['loss', 'val_loss', 'acc', 'val_acc'],
            { callbacks: ['onEpochEnd'] }
            )
    })

    window.predict = (form) => {
        const pred = model.predict(tf.tensor([[Number(form.a.value), Number(form.b.value), Number(form.c.value), Number(form.d.value)]]));
        alert(`预测结果 ${IRIS_CLASSES[pred.argMax(1).dataSync()[0]]}`)
    }
}