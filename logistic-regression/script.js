import * as tfvis from '@tensorflow/tfjs-vis'
import * as tf from '@tensorflow/tfjs'
import { getData } from './data'

window.onload = async () => {
    // 构建数据
    const data = getData(400)

    // 可视化
    tfvis.render.scatterplot(
        { name: '逻辑回归散点图' },
        { values: [
            data.filter(p => p.label === 1),
                data.filter(p => p.label === 0)
            ] }
    );

    // 初始化模型
    const model = tf.sequential();
    model.add(tf.layers.dense({
        units: 1,
        inputShape: [2],
        activation: 'sigmoid'
    }));

    model.compile({
            loss: tf.losses.logLoss,
            optimizer: tf.train.adam(0.1)
        });

    const inputs = tf.tensor(data.map(p => [p.x, p.y]));
    const labels = tf.tensor(data.map(p => p.label));

    await model.fit(inputs, labels, {
        batchSize: 40,
        epochs: 20,
        callbacks: tfvis.show.fitCallbacks(
            {name: '逻辑回归训练',},
            ['loss'])
    });

    window.predict = (form) => {
        const pred = model.predict(tf.tensor([[Number(form.x.value), Number(form.y.value)]]));
        alert(`预测结果 ${pred.dataSync()[0]}`)
    }

};
