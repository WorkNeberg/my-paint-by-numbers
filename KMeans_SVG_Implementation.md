# K-Means Algorithm and SVG Generation

This document provides an overview of the k-means algorithm and SVG generation, along with instructions for using these algorithms in your application.

## K-Means Algorithm

The k-means algorithm is used to cluster colors in images. Below are the main components and methods for implementing the k-means algorithm.

### Vector Class

The `Vector` class represents a point in the color space. It includes methods to calculate distances and averages.

```typescript
import { Random } from "../random";

export class Vector {

    public tag:any;
    
    constructor(public values: number[], public weight: number = 1) { }

    public distanceTo(p: Vector): number {
        let sumSquares = 0;
        for (let i: number = 0; i < this.values.length; i++) {
            sumSquares += (p.values[i] - this.values[i]) * (p.values[i] - this.values[i]);
        }

        return Math.sqrt(sumSquares);
    }

    /**
     *  Calculates the weighted average of the given points
     */
    public static average(pts: Vector[]): Vector {
        if (pts.length === 0) {
            throw Error("Can't average 0 elements");
        }

        const dims = pts[0].values.length;
        const values = [];
        for (let i: number = 0; i < dims; i++) {
            values.push(0);
        }

        let weightSum = 0;
        for (const p of pts) {
            weightSum += p.weight;

            for (let i: number = 0; i < dims; i++) {
                values[i] += p.weight * p.values[i];
            }
        }

        for (let i: number = 0; i < values.length; i++) {
            values[i] /= weightSum;
        }

        return new Vector(values);
    }
}
```

### KMeans Class

The `KMeans` class manages the clustering process with methods to initialize centroids and perform clustering steps.

```typescript
export class KMeans {

    public currentIteration: number = 0;
    public pointsPerCategory: Vector[][] = [];

    public centroids: Vector[] = [];
    public currentDeltaDistanceDifference: number = 0;

    constructor(private points: Vector[], public k: number, private random:Random, centroids: Vector[] | null = null) {

        if (centroids != null) {
            this.centroids = centroids;
            for (let i: number = 0; i < this.k; i++) {
                this.pointsPerCategory.push([]);
            }
        } else {
            this.initCentroids();
        }
    }

    private initCentroids() {
        for (let i: number = 0; i < this.k; i++) {
            this.centroids.push(this.points[Math.floor(this.points.length * this.random.next())]);
            this.pointsPerCategory.push([]);
        }
    }

    public step() {
        // clear category
        for (let i: number = 0; i < this.k; i++) {
            this.pointsPerCategory[i] = [];
        }

        // calculate points per centroid
        for (const p of this.points) {
            let minDist = Number.MAX_VALUE;
            let centroidIndex: number = -1;
            for (let k: number = 0; k < this.k; k++) {
                const dist = this.centroids[k].distanceTo(p);
                if (dist < minDist) {
                    centroidIndex = k;
                    minDist = dist;

                }
            }
            this.pointsPerCategory[centroidIndex].push(p);
        }

        let totalDistanceDiff = 0;

        // adjust centroids
        for (let k: number = 0; k < this.pointsPerCategory.length; k++) {
            const cat = this.pointsPerCategory[k];
            if (cat.length > 0) {
                const avg = Vector.average(cat);

                const dist = this.centroids[k].distanceTo(avg);
                totalDistanceDiff += dist;
                this.centroids[k] = avg;
            }
        }
        this.currentDeltaDistanceDifference = totalDistanceDiff;
    }
}
```

### Color Space Conversion

The code converts RGB colors to different color spaces (HSL, LAB) for clustering.

```javascript
let data;
if (settings.kMeansClusteringColorSpace === settings_1.ClusteringColorSpace.RGB) {
    data = rgb;
}
else if (settings.kMeansClusteringColorSpace === settings_1.ClusteringColorSpace.HSL) {
    data = colorconversion_1.rgbToHsl(rgb[0], rgb[1], rgb[2]);
}
else if (settings.kMeansClusteringColorSpace === settings_1.ClusteringColorSpace.LAB) {
    data = colorconversion_1.rgb2lab(rgb);
}
else {
    data = rgb;
}
```

### Clustering Process

The clustering process iteratively adjusts centroids based on the distance of points until a minimum delta distance difference is achieved.

```javascript
// determine the weight (#pointsOfColor / #totalpoints) of each color
const weight = pointsByColor[color].length / (imgData.width * imgData.height);
const vec = new clustering_1.Vector(data, weight);
vec.tag = rgb;
vectors[vIdx++] = vec;

const random = new random_1.Random(settings.randomSeed);
// vectors of all the unique colors are built, time to cluster them
const kmeans = new clustering_1.KMeans(vectors, settings.kMeansNrOfClusters, random);
let curTime = new Date().getTime();
kmeans.step();
while (kmeans.currentDeltaDistanceDifference > settings.kMeansMinDeltaDifference) {
    kmeans.step();
    // update GUI every 500ms
    if (new Date().getTime() - curTime > 500) {
        curTime = new Date().getTime();
        yield common_1.delay(0);
        if (onUpdate != null) {
            ColorReducer.updateKmeansOutputImageData(kmeans, settings, pointsByColor, imgData, outputImgData, false);
            onUpdate(kmeans);
        }
    }
}
// update the output image data (because it will be used for further processing)
ColorReducer.updateKmeansOutputImageData(kmeans, settings, pointsByColor, imgData, outputImgData, true);
if (onUpdate != null) {
    onUpdate(kmeans);
}
```

## SVG Generation

The SVG generation is part of the steps involved in creating the final vectorized images.

```html
<div class="col s2">
    <div class="status SVGGenerate">
        SVG generation
        <div class="progress">
            <div id="statusSVGGenerate" class="determinate" style="width: 0%"></div>
        </div>
    </div>
</div>
```

## Using the Algorithms in Your Application

To use the k-means algorithm and SVG generation in your application, follow these steps:

1. **Include the necessary classes and methods**: Add the `Vector`, `KMeans`, and color space conversion methods to your project.
2. **Initialize the k-means algorithm**: Create instances of `Vector` for your data points and initialize a `KMeans` instance.
3. **Perform clustering**: Call the `step` method of the `KMeans` instance iteratively until the desired delta distance difference is achieved.
4. **Generate SVG**: Use the results from the k-means clustering to generate the SVG output.

### Example Usage

```typescript
import { Vector, KMeans } from './path_to_your_kmeans_module';
import { Random } from './path_to_your_random_module';

// Create data points (example)
const points = [
    new Vector([255, 0, 0]),
    new Vector([0, 255, 0]),
    new Vector([0, 0, 255])
];

// Initialize KMeans
const random = new Random(12345);
const kmeans = new KMeans(points, 3, random);

// Perform clustering
while (kmeans.currentDeltaDistanceDifference > 1) {
    kmeans.step();
}

// Use clustering results for SVG generation
// (Implement your SVG generation logic here)
```

By following these steps, you can integrate the k-means algorithm and SVG generation into your application.
