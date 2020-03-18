// import * as tfvis from '@tensorflow/tfjs-vis'
// import * as tf from '@tensorflow/tfjs'
import { getIrisData, IRIS_CLASSES, IRIS_NUM_CLASSES } from './data.js'

window.onload = () => {
    console.log(IRIS_CLASSES)
    console.log(IRIS_NUM_CLASSES);
    const [xTrain, yTrain, xTest, yTest] = getIrisData(0.15);
    xTrain.print();
    yTrain.print();
    xTest.print();
    yTest.print();
}