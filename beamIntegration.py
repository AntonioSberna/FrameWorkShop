'''
numerical integration for element stifness and element forces
added: GaussLegendre, GaussLobatto
to be added: GaussRadau, Modified GaussRadau2 (requires the length of the element as input)
'''
class BeamIntegration:
    def __init__(
            self,
            id: str | int,
            integration_type: str, # integration type
            nIP: int               # number of Gauss Integration Point
    ) -> None:
        
        self.id  = id,
        self.integration_type = integration_type.lower()
        self.nIP = nIP

        if type(self.nIP) is not int:
            raise ValueError(f"number of GaussIP must be an integer")

        self.position = None
        self.weight   = None

        match self.integration_type:

            case "legendre":   
                self._gauss_legendre()
        
            case "lobatto":         
                self._gauss_lobatto()

            case _:
                raise ValueError(f"Unknown integration type: {self.integration_type}")

        if self.position is None or self.weight is None:
            raise ValueError(f"Unsupported number of integration points: {self.nIP}")
        
    def _gauss_legendre(self):        
        match self.nIP:

            case 1:
                self.position = [0.0]
                self.weight = [2.0]
            
            case 2:
                self.position = [-0.5773502691896257, 0.5773502691896257]
                self.weight = [1.0, 1.0]

            case 3:
                self.position = [-0.7745966692414834, 0.0, 0.7745966692414834]
                self.weight = [0.5555555555555558, 0.8888888888888883, 0.5555555555555558]

            case 4:
                self.position = [-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526]
                self.weight = [0.3478548451374536, 0.6521451548625464, 0.6521451548625464, 0.3478548451374536]

            case 5:
                self.position = [-0.9061798459386641, -0.5384693101056831, 0.0, 0.5384693101056831, 0.9061798459386641]
                self.weight = [0.23692688505618908, 0.47862867049936647, 0.5688888888888889, 0.47862867049936647, 0.23692688505618908]

            case 6:
                self.position = [-0.932469514203152, -0.6612093864662645, -0.23861918608319696, 0.23861918608319696, 0.6612093864662645, 0.932469514203152]
                self.weight = [0.17132449237917022, 0.36076157304813866, 0.46791393457269104, 0.46791393457269104, 0.36076157304813866, 0.17132449237917022]

            case 7:
                self.position = [-0.9491079123427585, -0.7415311855993945, -0.40584515137739713, 0.0, 0.40584515137739713, 0.7415311855993945, 0.9491079123427585]
                self.weight = [0.12948496616887006, 0.2797053914892765, 0.3818300505051188, 0.4179591836734691, 0.3818300505051188, 0.2797053914892765, 0.12948496616887006]

            case 8:
                self.position = [-0.9602898564975362, -0.7966664774136267, -0.525532409916329, -0.18343464249564984, 0.18343464249564984, 0.525532409916329, 0.7966664774136267, 0.9602898564975362]
                self.weight = [0.1012285362903762, 0.22238103445337487, 0.3137066458778872, 0.3626837833783617, 0.3626837833783617, 0.3137066458778872, 0.22238103445337487, 0.1012285362903762]

            case _:
                raise ValueError("max 8 GaussLegendre IP")

        ''' ADD INPUT: LENGTH [0,1] or [0,L]
        # calculate positions and weights along the length of the element
        nIP = len(self.position)
        for i in range(nIP):
            self.position[i] = (self.position[i]+1)/2 * self.L 
            self.weight[i] = self.weight[i]*0.5
        '''

            
    def _gauss_lobatto(self):
        match self.nIP:

            case 1:
                raise ValueError("min 2 GaussLobatto IP")

            case 2:
                self.position = [-1.0, 1.0]
                self.weight = [1.0, 1.0]

            case 3:
                self.position = [-1.0, 0.0, 1.0]
                self.weight = [1/3, 4/3, 1/3]

            case 4:
                self.position = [-1.0, -0.4472135954999579, 0.4472135954999579, 1.0]
                self.weight = [1/6, 5/6, 5/6, 1/6]

            case 5:
                self.position = [-1.0, -0.6546536707079772, 0.0, 0.6546536707079772, 1.0]
                self.weight = [0.1, 0.5444444444444444, 0.7111111111111111, 0.5444444444444444, 0.1]

            case 6:
                self.position = [-1.0, -0.7650553239294647, -0.2852315164806451, 0.2852315164806451, 0.7650553239294647, 1.0]
                self.weight = [0.0666666666666667, 0.3784749562978470, 0.5548583770354863, 0.5548583770354863, 0.3784749562978470, 0.0666666666666667]

            case 7:
                self.position = [-1.0, -0.8302238962785669, -0.4688487934707142, 0.0, 0.4688487934707142, 0.8302238962785669, 1.0]
                self.weight = [0.0476190476190476, 0.276826047361566, 0.4317453812098627, 0.4876190476190476, 0.4317453812098627, 0.276826047361566, 0.0476190476190476]

            case 8:
                self.position = [-1.0, -0.8717401485096066, -0.5917001814331423, -0.2092992179024789, 0.2092992179024789, 0.5917001814331423, 0.8717401485096066, 1.0]
                self.weight = [0.0357142857142857, 0.2107042271435061, 0.3411226924835044, 0.4124587946587038, 0.4124587946587038, 0.3411226924835044, 0.2107042271435061, 0.0357142857142857]

            case _:
                raise ValueError("max 8 GaussLobatto IP")

        ''' ADD INPUT: LENGTH [0,1] or [0,L]
        # calculate positions and weights along the length of the element
        nIP = len(self.position)
        for i in range(nIP):
            self.position[i] = (self.position[i]+1)/2 * self.L 
            self.weight[i] = self.weight[i]*0.5
        '''


    def get_location_and_weight(self) -> tuple[list[float], list[float]]:

        xi = self.position
        wt = self.weight
        return xi, wt

# test

for intType in ["Legendre","Lobatto"]:
    for nIP in [2,3,4,5,6,7,8]:
        print("-"*50)
        print("Integration type: ",intType,"  nIP: ",nIP)
        aa = BeamIntegration(1,intType,nIP)
        print(aa.get_location_and_weight())