module HandWrittenRecog(
    eval,
    bestOf,
    getwsbs
)where

import Prelude
import System.Random
import System.IO
import System.Environment
import Data.List.Split
import Control.Monad.State as Mod
import Data.Maybe
import Data.Vector as V
import Data.Matrix as M
import Data.ByteString.Lazy as BS
import Data.Int
import Codec.Compression.GZip (decompress)
import Data.List as L
import Data.Ord

mean::V.Vector Double->Double
mean xs=(/) (V.sum xs)  len
    where len=fromIntegral (V.length xs)

matSum::Matrix Double ->Int ->Double
matSum xs i = 
    if i==((M.nrows xs) +1) 
    then V.sum  ( M.getRow 1 xs)
    else matSum (M.combineRows 1 1 i xs) (i+1)

matMean::Matrix Double->Double
matMean xs =  (/) ((matSum xs 1)) (fromIntegral ((*) (M.ncols xs)  (M.nrows xs)))

matMax::Matrix Double->Double->Int->Double
matMax xs tempmax i
    |(i==(M.nrows xs))=tempmax
    |otherwise = matMax xs (max tempmax (V.maximum (M.getRow i xs))) (i+1)
 
layerCom :: Matrix Double -> V.Vector Double -> V.Vector Double ->V.Vector Double
layerCom ws bias  xs =M.getRow 1 ((+) ((M.rowVector xs) `multStd` ws) (M.rowVector bias))
 
softmax :: V.Vector Double -> V.Vector Double
softmax vec = V.map (\x -> exp x / V.sum (V.map exp safeVec)) safeVec
  where
    safeVec = maxSafe $ minusMean vec
    minusMean v = V.map (\x -> x - mean v) v
    maxSafe v 
        |(V.maximum v) > 500 = maxSafe (V.map (flip (-) 500) v)
        |otherwise = v
vectorDot :: V.Vector Double -> V.Vector Double -> Double
vectorDot vec1 vec2 = V.sum $ V.zipWith (*) vec1 vec2
crossEntropy :: V.Vector Double -> V.Vector Double -> Double
crossEntropy real  hypo = -(vectorDot real $ V.map log $ check hypo)
  where
    check v = V.map (max 1.0e-300) v
matMap :: (a -> a) -> Matrix a -> Matrix a
matMap f m = matrix (M.nrows m) (M.ncols m) (\(i,j)->f $ M.getElem i j m)
 
diffWsMat3 :: V.Vector Double -> V.Vector Double -> V.Vector Double ->V.Vector Double -> Matrix Double
diffWsMat3 ts s_m f2 relu3= (colVector ( f2)) `multStd` (rowVector $ V.zipWith (-) (V.zipWith (*) relu3 s_m) ts)
 
diffBiasVec3 :: V.Vector Double -> V.Vector Double ->V.Vector Double -> V.Vector Double
diffBiasVec3 ts s_m relu3= V.zipWith (-) (V.zipWith (*) relu3 s_m) ts

diffWsMat2 :: V.Vector Double -> V.Vector Double -> V.Vector Double -> Matrix Double -> V.Vector Double ->V.Vector Double ->Matrix Double
diffWsMat2 ts s_m f1 w3 relu2 relu3= (colVector ( f1)) `multStd`  (rowVector $ V.zipWith (*) relu2 $ getRow 1 ((rowVector $ V.zipWith3 (\r s t->r*s-t)  relu3 s_m ts) `multStd` (M.transpose w3)))

diffBiasVec2 :: V.Vector Double -> V.Vector Double ->Matrix Double ->V.Vector Double ->  V.Vector Double ->V.Vector Double
diffBiasVec2 ts s_m w3 relu2 relu3= V.zipWith (*) relu2 $ getRow 1 ((rowVector $ V.zipWith3 (\r s t->r*s-t)  relu3 s_m ts) `multStd`  (M.transpose w3))

diffWsMat1 :: V.Vector Double -> V.Vector Double -> V.Vector Double -> Matrix Double -> Matrix Double->V.Vector Double -> V.Vector Double ->V.Vector Double -> Matrix Double
diffWsMat1 ts s_m input w3 w2 relu1 relu2 relu3=  let 
        s=(rowVector $ V.zipWith3 (\r s t->r*s-t) relu3 s_m ts)
        w3T=(rowVector $ V.zipWith (*) relu2 $ getRow 1 (s `multStd` (M.transpose w3)))
        w2T=M.transpose w2
        in (colVector input) `multStd`  (rowVector $ V.zipWith (*) relu1 $ getRow 1 (w3T `multStd` w2T))
diffBiasVec1 :: V.Vector Double -> V.Vector Double -> Matrix Double -> Matrix Double ->V.Vector Double -> V.Vector Double -> V.Vector Double ->(V.Vector Double)
diffBiasVec1 ts s_m w3 w2 relu1 relu2 relu3= let 
        s=(rowVector $ V.zipWith3 (\r s t->r*s-t)  relu3 s_m ts)
        w3T=(rowVector $ V.zipWith (*) relu2 $ getRow 1 (s `multStd` (M.transpose w3)))
        w2T=M.transpose w2
        db1=V.zipWith (*) relu1 $ getRow 1 (w3T `multStd` w2T)
        in db1
gradientDescent_one :: V.Vector Double->([Matrix Double],[V.Vector Double])->Double->[V.Vector Double]-> ([Matrix Double], [V.Vector Double])
gradientDescent_one ts (weights,biases) alpha  feats =
  let
    trainedWs3 =if matMax dWsMat3 0 1 < 0.05  then (weights!!2) else (-) (weights!!2)  (M.scaleMatrix alpha dWsMat3)
    trainedBias3 =if  V.maximum dBiasVec3 < 0.05 then (biases!!2) else V.zipWith (-) (biases!!2) $ V.map (*alpha) dBiasVec3
    trainedWs2 =if matMax dWsMat2 0 1 < 0.05  then (weights!!1) else  (-) (weights!!1)  (M.scaleMatrix alpha dWsMat2)
    trainedBias2 =if  V.maximum dBiasVec2 < 0.05 then (biases!!1) else V.zipWith (-) (biases!!1) $ V.map (*alpha) dBiasVec2
    trainedWs1 =if matMax dWsMat1 0 1 < 0.05  then (weights!!0) else  (-) (weights!!0)  (M.scaleMatrix alpha dWsMat1)
    trainedBias1 =if  V.maximum dBiasVec1 < 0.05 then (biases!!0) else V.zipWith (-) (biases!!0) $ V.map (*alpha) dBiasVec1
    dWsMat3 = diffWsMat3 ts (feats!!4) (feats!!2) (relu' (feats!!3))
    dBiasVec3 = diffBiasVec3 ts (feats!!4) (relu' (feats!!3))
    dWsMat2 = diffWsMat2 ts (feats!!4) (feats!!1) (weights!!2) (relu' (feats!!2)) (relu' (feats!!3))
    dBiasVec2 = diffBiasVec2 ts (feats!!4) (weights!!2)  (relu' (feats!!2)) (relu' (feats!!3))
    dWsMat1 = diffWsMat1 ts (feats!!4) (feats!!0) (weights!!2) (weights!!1) (relu' (feats!!1)) (relu' (feats!!2)) (relu' (feats!!3))
    dBiasVec1 = diffBiasVec1 ts (feats!!4) (weights!!2) (weights!!1) (relu' (feats!!1)) (relu' (feats!!2)) (relu' (feats!!3))
    in ([trainedWs1,trainedWs2,trainedWs3], [trainedBias1,trainedBias2,trainedBias3])
relu::V.Vector Double->V.Vector Double
relu xs=V.map (max 0) xs
relu'::V.Vector Double->V.Vector Double
relu' =V.map (\x->if x>0 then 1 else 0)
fcBias::Int->V.Vector Double
fcBias row = V.replicate row 0
forward::V.Vector Double->([Matrix Double],[V.Vector Double])->[V.Vector Double]
forward image (weights,biases)= let
          hypo=softmax feat3
          feat3=relu $ layerCom (weights!!2) (biases!!2) feat2
          feat2=relu $ layerCom (weights!!1) (biases!!1) feat1
          feat1=relu $ layerCom (weights!!0) (biases!!0) image
          in [feat1,feat2,feat3,hypo]
alpha::Double
alpha= 0.002
getImage s n = fromIntegral . BS.index s . (n*28^2 + 16 +) <$> [0..28^2 - 1] 
getX     s n = (/ 256) <$> getImage s n
getLabel s n = fromIntegral $ BS.index s (n + 8)
getY     s n = fromIntegral . fromEnum . (getLabel s n ==) <$> [0..9]
learn:: V.Vector Double->V.Vector Double->([Matrix Double],[V.Vector Double])->([Matrix Double],[V.Vector Double])
learn image ts (weights,biases)= 
    let (tws,tbs)=gradientDescent_one ts (weights,biases) alpha feats
        feats=[image]  Prelude.++ (forward image (weights,biases))
        loss=crossEntropy ts (feats!!4)
        in (tws,tbs)
eval::V.Vector Double->([Matrix Double],[V.Vector Double])->[Double]
eval image (weights,biases)= let
          hypo=softmax feat3
          feat3=relu $ layerCom (weights!!2) (biases!!2) feat2
          feat2=relu $ layerCom (weights!!1) (biases!!1) feat1
          feat1=relu $ layerCom (weights!!0) (biases!!0) image
          in V.toList hypo
save_txt::FilePath 
save_txt = "/home/jxf/haskell/mnist_only_base/trained_fc_network_6w.txt"
conventor::[[String]]->[[Double]]
conventor = Prelude.map $ Prelude.map (\x ->read x::Double)
getwsbs ::String->([Matrix Double],[V.Vector Double])
getwsbs trainwsbs=
    let 
        wb_all=L.drop 1 $ splitOneOf "[" trainwsbs
        wb_1=L.map (\xs->L.delete ']' xs) wb_all 
        wb_2=L.map (\xs->splitOn "," xs) wb_1
        wb_3= conventor wb_2
        w1=M.fromList 784 256 $ wb_3!!0
        w2=M.fromList 256 256 $ wb_3!!1
        w3=M.fromList 256 10 $ wb_3!!2
        b1=V.fromList $ wb_3!!3
        b2=V.fromList $ wb_3!!4
        b3=V.fromList $ wb_3!!5
    in ([w1,w2,w3],[b1,b2,b3])
render n = let s = " .:oO@" in s !! (fromIntegral n * L.length s `div` 256)
bestOf = fst . L.maximumBy (comparing snd) . L.zip [0..]
train_scheme::IO()
train_scheme = do 
    [trainI, trainL, testI, testL] <- Mod.mapM ((decompress  <$>) . BS.readFile) [ "train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz", "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]
    wsbs_1w <-System.IO.readFile  "./trained_fc_network.txt"
    let w1=M.fromList 784 256 (randomRs (-0.1,0.1) (mkStdGen 1)::[Double])
        b1=fcBias 256
        w2=M.fromList 256 256 (randomRs (-0.1,0.1) (mkStdGen 2)::[Double])
        b2=fcBias 256
        w3=M.fromList 256 10 (randomRs (-0.1,0.1) (mkStdGen 3)::[Double])
        b3=fcBias 10
        iter_total=10000
        wsbs=getwsbs wsbs_1w
    n <- (`mod` 10000) <$> randomIO
    System.IO.putStr . L.unlines $ L.take 28 $ L.take 28 <$> L.iterate (L.drop 28) (render <$> getImage testI n) --生成字符图像

    let example = V.fromList $ getX testI n
        trainwsbs = L.scanl (L.foldl' (\wsbs_cur index -> learn (V.fromList $ getX trainI index) (V.fromList $ getY trainL index) wsbs_cur)) wsbs [[0.. 999],[1000.. 3999],[4000.. 6999],[7000..9999]]

        smart = L.last trainwsbs
        cute d score = show d L.++ ": " L.++ L.replicate (round $ 70 * min 1 score) '+'
    Mod.forM_ trainwsbs $ System.IO.putStrLn . L.unlines . L.zipWith cute [0..9] . eval example
    System.IO.putStrLn $ "best guess: " L.++ show (bestOf $ eval example  smart)
    let guesses = bestOf . (\n -> eval (V.fromList $ getX testI n) smart) <$> [0..9999]
    let answers = getLabel testL <$> [0..9999]
    System.IO.putStrLn $ show (L.sum $ fromEnum <$> L.zipWith (==) guesses answers) L.++ " / 10000"
    let (trainws,trainbs)=smart
    System.IO.writeFile save_txt $ show $ M.toList $ trainws!!0
    System.IO.appendFile save_txt $ show $ M.toList $ trainws!!1
    System.IO.appendFile save_txt $ show $ M.toList $  trainws!!2
    System.IO.appendFile save_txt $ show $ V.toList $ trainbs!!0
    System.IO.appendFile save_txt $ show $ V.toList $ trainbs!!1
    System.IO.appendFile save_txt $ show $ V.toList $ trainbs!!2




