import * as tfvis from '@tensorflow/tfjs-vis'
import * as tf from '@tensorflow/tfjs'

window.onload = async () => {
    const heights = [150, 160, 170];
    const weights = [40, 50, 60];

    tfvis.render.scatterplot(
        {name: '身高体重训练数据'},
        {values: heights.map((x, i) => ({
                x, y: weights[i]
            })) },
        { xAxisDomain: [140, 200], yAxisDomain: [30, 70]}
    );

    // 定义模型结构
    const model = tf.sequential();

    // 添加层数
    model.add(tf.layers.dense({
        units: 1,
        inputShape: [1],
    }));

    // 设置损失函数和优化器
    model.compile({
        loss: tf.losses.meanSquaredError,
        optimizer: tf.train.sgd(0.1),
    });

    // 设置归一化数据
    const input = tf.tensor(heights).sub(150).div(20);
    const label = tf.tensor(weights).sub(40).div(20);

    // 训练过程，异步的过程
    await model.fit(input, label, {
        batchSize: 3,
        epochs: 200,
        callbacks: tfvis.show.fitCallbacks(
            {name: '梯度调参',},
            ['loss'])
    });

    const output = model.predict(tf.tensor([180]).sub(150).div(20));
    output.print();
    console.log(output.mul(20).add(40).dataSync())
}