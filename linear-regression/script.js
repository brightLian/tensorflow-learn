import * as tfvis from '@tensorflow/tfjs-vis'
import * as tf from '@tensorflow/tfjs'

window.onload = async () => {
    const xs = [1, 2, 3, 4];
    const ys = [1, 3, 5, 7];

    tfvis.render.scatterplot(
        { name: '样本' },
        { values: [xs.map((x, i) => ({x, y: ys[i]}))]},
        { xAxisDomain: [0, 5], yAxisDomain: [0, 8]}
        )

    const model = tf.sequential();
    // 添加层数
    model.add(tf.layers.dense({
        units: 1,
        inputShape: [1],
    }));
    // 设置损失函数和优化器
    model.compile({
        loss: tf.losses.meanSquaredError,
        optimizer: tf.train.sgd(0.11),
    });
    const input = tf.tensor(xs);
    const label = tf.tensor(ys);
    // 训练过程，异步的过程
    await model.fit(input, label, {
        batchSize: 4,
        epochs: 200,
        callbacks: tfvis.show.fitCallbacks(
            {name: '梯度调参',},
            ['loss'])
    });

    const output = model.predict(tf.tensor([5]));
    output.print();
    console.log(output.dataSync())
};
