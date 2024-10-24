# CHANGELOG



## v0.6.4 (2024-10-24)

### Fix

* fix: Merge pull request #33 from jbloomAus/fix/topk-selection-purview

Fix/topk selection purview ([`afccd5a`](https://github.com/jbloomAus/SAEDashboard/commit/afccd5aaa00d00672eb1270b258b69f0e51c046a))

### Unknown

* Update README.md ([`8235a9e`](https://github.com/jbloomAus/SAEDashboard/commit/8235a9e3adaea50b6b9f26f575e25a254d67a135))

* updated formatting/typing ([`fb141ae`](https://github.com/jbloomAus/SAEDashboard/commit/fb141ae991261408d296286bf6777b2ec5f1f319))

* Merge pull request #32 from jbloomAus/docs/readme-update

docs: updated readme ([`b5e5480`](https://github.com/jbloomAus/SAEDashboard/commit/b5e54808ee05fc75e68d74ec319bf49826b45508))

* TopK will now select from all latents regardless of feature batch size ([`c1f0e14`](https://github.com/jbloomAus/SAEDashboard/commit/c1f0e14dda7aa3364bfd78ca2b8c04c95b2d14b3))


## v0.6.3 (2024-10-23)

### Fix

* fix: update cached_activations directory to include number of prompts ([`0308cb1`](https://github.com/jbloomAus/SAEDashboard/commit/0308cb146bf2eb9cee26f03d3098511d03022485))


## v0.6.2 (2024-10-23)

### Fix

* fix: lint ([`3fc0e2c`](https://github.com/jbloomAus/SAEDashboard/commit/3fc0e2ccb39ed1d3e31d66ae0aba2b2b367d46aa))

### Unknown

* Merge branch &#39;main&#39; of https://github.com/jbloomAus/SAEDashboard ([`8f74a96`](https://github.com/jbloomAus/SAEDashboard/commit/8f74a969f48a7e0fd8de17cc983acf3886db95ef))

* Fix: divide by zero, cached_activations folder name ([`1792298`](https://github.com/jbloomAus/SAEDashboard/commit/179229805ae6489d86e235240c65d26db64b5cd7))


## v0.6.1 (2024-10-22)

### Fix

* fix: update saelens to v4 ([`ef1a330`](https://github.com/jbloomAus/SAEDashboard/commit/ef1a3302d0483eddb247defab5c88816850f7f63))

### Unknown

* Merge branch &#39;main&#39; of https://github.com/jbloomAus/SAEDashboard ([`508a74d`](https://github.com/jbloomAus/SAEDashboard/commit/508a74df8ff279716501e4179c501b5089a8d706))


## v0.6.0 (2024-10-21)

### Feature

* feat: np sae id suffix ([`448b14e`](https://github.com/jbloomAus/SAEDashboard/commit/448b14e0b3aea8ff854a5365f164b6ce5f419f0d))

### Unknown

* Update README.md ([`a1546fd`](https://github.com/jbloomAus/SAEDashboard/commit/a1546fdef32745cdc862a5a2dd0478e57e45320d))

* Removed outdated vis type ([`b0676af`](https://github.com/jbloomAus/SAEDashboard/commit/b0676afcca0845b73a54d983eaa9d72b0e9dff05))

* Update README.md ([`9b8446a`](https://github.com/jbloomAus/SAEDashboard/commit/9b8446aa47f287ba80bf0ac4a39f7c77f0492990))

* Updated format ([`90e4a09`](https://github.com/jbloomAus/SAEDashboard/commit/90e4a09eedd7f428b64e58d5ca2fd1cfa658b0da))

* Updated readme ([`f6819a6`](https://github.com/jbloomAus/SAEDashboard/commit/f6819a6da594673cad65c9ccd3a4f67746de796d))

* Merge pull request #31 from jbloomAus/fix/reduce-mem

fix: added mem cleanup ([`60bd716`](https://github.com/jbloomAus/SAEDashboard/commit/60bd716c7b52bb0eaea0937e097eb77ed78bd33d))

* Fixed formatting ([`f1fab0c`](https://github.com/jbloomAus/SAEDashboard/commit/f1fab0c1fd5be281e2162ab3f54ffc7f4c09a1ce))

* Added cleanup ([`305c46d`](https://github.com/jbloomAus/SAEDashboard/commit/305c46d7a30330bbae6893b83cb6d498c2c975f1))

* Merge pull request #30 from jbloomAus/feat-mask-via-position

feat: prepending/appending tokens for prompt template + feat mask via Position ([`4c60e4c`](https://github.com/jbloomAus/SAEDashboard/commit/4c60e4c834dfb5759ce55dc90d1f88768abfea0d))

* add a few tests ([`96247d5`](https://github.com/jbloomAus/SAEDashboard/commit/96247d5afaf141b8b1279c17fd135240b0d8e869))

* handle prefixes / suffixes and ignored positions ([`bff7fd9`](https://github.com/jbloomAus/SAEDashboard/commit/bff7fd98b09318a1b01d2bc4a06467f8afa156f9))

* simplify masking ([`385b6e1`](https://github.com/jbloomAus/SAEDashboard/commit/385b6e116ecac53ad4df8585f7513c3416707d8b))

* add option for ignoring tokens at particular positions ([`ed3426d`](https://github.com/jbloomAus/SAEDashboard/commit/ed3426de5cb1495c138f770eefa5f941408aa390))

* Merge pull request #29 from jbloomAus/refactor/optimize-dfa-speed

Sped up DFA calculation 60x ([`f992e3c`](https://github.com/jbloomAus/SAEDashboard/commit/f992e3cf116189625b3a92529cf68d6226a1221c))

* Sped up DFA calculation ([`be11cd5`](https://github.com/jbloomAus/SAEDashboard/commit/be11cd5652f0f8a8ae425555666b747b9b99314e))

* Added test to check for decoder weight dist (head dist) ([`f147696`](https://github.com/jbloomAus/SAEDashboard/commit/f1476967af5fee95313264ccaee668605d23b9ad))

* Merge pull request #28 from jbloomAus/feature/np-topk-size-arg

Feature/np topk size arg ([`c5c1365`](https://github.com/jbloomAus/SAEDashboard/commit/c5c136576609991177d3a8924b5bf75a42b66399))

* Merge pull request #25 from jbloomAus/fix/dfa-for-gqa

Fix/dfa for gqa ([`85c345f`](https://github.com/jbloomAus/SAEDashboard/commit/85c345f3ad8069a59be8d495242395c50381ab01))

* Fixed formatting ([`48a67c7`](https://github.com/jbloomAus/SAEDashboard/commit/48a67c79247d05745d355e6a4bf380e9df20474e))

* Removed redundant code from rebase ([`a71fb9d`](https://github.com/jbloomAus/SAEDashboard/commit/a71fb9dde6e880b0f4297277d27696c9d524d052))

* fixed rebase ([`57ee280`](https://github.com/jbloomAus/SAEDashboard/commit/57ee28021efd3678bcd9d12d55e048c14a2f2d47))

* Added tests for DFA for GQA ([`fcfac37`](https://github.com/jbloomAus/SAEDashboard/commit/fcfac37e148461e585f38fddf868ad2a32d908a8))

* Removed duplicate code ([`cc00944`](https://github.com/jbloomAus/SAEDashboard/commit/cc00944855720d5b8139d4267b44c1a230ef5319))

* Fixed formatting ([`50b08b4`](https://github.com/jbloomAus/SAEDashboard/commit/50b08b4eb50734afe0f085274ccaee71ec4017a4))

* Removed debugging statements ([`f7b949b`](https://github.com/jbloomAus/SAEDashboard/commit/f7b949b4af6bc8ca7557bfa5fa2441fbaa0284a0))

* more debug prints x3 ([`53536b0`](https://github.com/jbloomAus/SAEDashboard/commit/53536b03d624783b6b2f95b07b9318139ef0c49e))

* more debug prints x2 ([`6f2c504`](https://github.com/jbloomAus/SAEDashboard/commit/6f2c504a355f9071e766fc7fa3b6aad9890572a8))

* more debug prints ([`e1bef90`](https://github.com/jbloomAus/SAEDashboard/commit/e1bef90d16e8c73c9532b19a08c842757828c7ed))

* temp print statements ([`fd75714`](https://github.com/jbloomAus/SAEDashboard/commit/fd75714ee4631463c1f754d68f83b9ef75eb2285))

* updated ignore ([`c01062f`](https://github.com/jbloomAus/SAEDashboard/commit/c01062faecfaa132d87c56a7ba7add573c6b0f4e))

* Reduced memory load of GQA DFA ([`1ae40e9`](https://github.com/jbloomAus/SAEDashboard/commit/1ae40e9d487af7e8a7b148629588ef87fdd0a6e5))

* DFA will now work for models with grouped query attention ([`c66c90f`](https://github.com/jbloomAus/SAEDashboard/commit/c66c90f5d51961cafd5f13c26a94193ee38f828a))

* Edited default chunk size ([`3c78bdc`](https://github.com/jbloomAus/SAEDashboard/commit/3c78bdcfda12e5873de082a7f1e631a801bd9407))

* Fixed formatting ([`10a36e3`](https://github.com/jbloomAus/SAEDashboard/commit/10a36e3e8da3c7593058d3638ac3b7a32953b1b0))

* Removed debugging statements and added device changes ([`0f51dd9`](https://github.com/jbloomAus/SAEDashboard/commit/0f51dd953cd214244c71e8b9156b90483ceaa2be))

* more debug prints x3 ([`112ef42`](https://github.com/jbloomAus/SAEDashboard/commit/112ef4292b81a64f6168e7527ec583faa9ba20a4))

* more debug prints x2 ([`ef154d6`](https://github.com/jbloomAus/SAEDashboard/commit/ef154d6044bb67d17a2aa225ddf4099ccfc16b55))

* more debug prints ([`1b18d14`](https://github.com/jbloomAus/SAEDashboard/commit/1b18d141dd33e3a99c2abd5a6d195ab5142890d8))

* temp print statements ([`2194d2c`](https://github.com/jbloomAus/SAEDashboard/commit/2194d2cea16856c96ace47ad5ac560f088e769b0))

* Lowered default threshold ([`a49d1e5`](https://github.com/jbloomAus/SAEDashboard/commit/a49d1e5b94c8ef680448f20ded849c7752fb5131))

* updated ignore ([`2067655`](https://github.com/jbloomAus/SAEDashboard/commit/20676554541d29fddd87215a47e8e94891e342ac))

* Reduced memory load of GQA DFA ([`8ec1956`](https://github.com/jbloomAus/SAEDashboard/commit/8ec19566e8898413d349fe3f2e43fbff232ffa62))

* DFA will now work for models with grouped query attention ([`8f3cf55`](https://github.com/jbloomAus/SAEDashboard/commit/8f3cf5532e57abc6e694fb11c5f9c7c2915215c0))

* Added head attr weights functionality for when DFA is use ([`234ea32`](https://github.com/jbloomAus/SAEDashboard/commit/234ea3211ce7dbf84d101c4e8bfe844c3903b16a))

* Added tests for DFA for GQA ([`3b99e36`](https://github.com/jbloomAus/SAEDashboard/commit/3b99e36c74d2c61617cfed107bee3b0eb3b63294))

* Simply updated default value for top K ([`5c855fe`](https://github.com/jbloomAus/SAEDashboard/commit/5c855fec0e58a114a537590d1400eaa42dd3610c))

* Testing variable topk sizes ([`79fe14b`](https://github.com/jbloomAus/SAEDashboard/commit/79fe14b840991bd1f8ada8462aeb65d72821c4aa))

* Merge pull request #27 from jbloomAus/fix/resolve-duplication

Removed sources of duplicate sequences ([`525bffe`](https://github.com/jbloomAus/SAEDashboard/commit/525bffee516a630c4b4f033d3971fad8c6dd5a74))

* Updated location of wandb finish() ([`921da77`](https://github.com/jbloomAus/SAEDashboard/commit/921da77132a560505fa61decf287ca3833f96ec7))

* Added two sets of tests for duplication checks ([`3e95ffd`](https://github.com/jbloomAus/SAEDashboard/commit/3e95ffd1dafd01deb1f7817845ccb6229fb4ae09))

* Restored original random indices function as it seemed ok ([`388719b`](https://github.com/jbloomAus/SAEDashboard/commit/388719bec99b4306e81e0cdb772b9924db210774))

* Removed sources of duplicate sequences ([`853306c`](https://github.com/jbloomAus/SAEDashboard/commit/853306c4e08d9ec95674fdc5c87f807019055d0d))

* Removed duplicate code ([`7093773`](https://github.com/jbloomAus/SAEDashboard/commit/7093773d079cd235aea99273a1365363a5bf8b6d))

* More rebasing stuff ([`59c6cd8`](https://github.com/jbloomAus/SAEDashboard/commit/59c6cd85ead287b2774aa591463d131840c7f270))

* Edited default chunk size ([`7d68f9e`](https://github.com/jbloomAus/SAEDashboard/commit/7d68f9e7131b8c5558e886022625dac267f20aab))

* Fixed formatting ([`4d5f38b`](https://github.com/jbloomAus/SAEDashboard/commit/4d5f38beca15f2ce05c89f83eb3e955c291f9687))

* Removed debugging statements and added device changes ([`76e17c9`](https://github.com/jbloomAus/SAEDashboard/commit/76e17c91a41b5df6047baa5bcfa33d253b029d29))

* more debug prints x3 ([`06535d3`](https://github.com/jbloomAus/SAEDashboard/commit/06535d3df168d92ac79d2f5a14b345c757dfd9de))

* more debug prints x2 ([`26e8297`](https://github.com/jbloomAus/SAEDashboard/commit/26e8297888de066f0097e3b73245eb149bfb327f))

* more debug prints ([`9ded356`](https://github.com/jbloomAus/SAEDashboard/commit/9ded356ea8c3c5dd841bf5a45ea65ae8c67935f5))

* temp print statements ([`024ad57`](https://github.com/jbloomAus/SAEDashboard/commit/024ad578b65b8f3592b42b66dc6a56aeae2a3116))

* Lowered default threshold ([`a3b5977`](https://github.com/jbloomAus/SAEDashboard/commit/a3b5977c0f1bb7a865f7349304a5dd8092f7c2e8))

* updated ignore ([`d5d325a`](https://github.com/jbloomAus/SAEDashboard/commit/d5d325a63b3b26b890c2bab512f2a8473bdc926a))

* Reduced memory load of GQA DFA ([`93eb1a9`](https://github.com/jbloomAus/SAEDashboard/commit/93eb1a9a92320d9f4645b500e22a566135918e3d))

* DFA will now work for models with grouped query attention ([`6594155`](https://github.com/jbloomAus/SAEDashboard/commit/65941559bac03a3e4fb128d5327033e01f19c18d))

* Added head attr weights functionality for when DFA is use ([`9312d90`](https://github.com/jbloomAus/SAEDashboard/commit/9312d901bf17e14400199c86e0284be6c750162a))


## v0.5.1 (2024-08-27)

### Fix

* fix: multi-gpu-tlens

fix: handle multiple tlens devices ([`ed1e967`](https://github.com/jbloomAus/SAEDashboard/commit/ed1e967d44b887f4b99d2257934ca920d5c6a508))

* fix: handle multiple tlens devices ([`ba5368f`](https://github.com/jbloomAus/SAEDashboard/commit/ba5368f9999f08332c153816ba5836f8a1eb9ba1))

### Unknown

* Fixed formatting ([`ed7d3b1`](https://github.com/jbloomAus/SAEDashboard/commit/ed7d3b16a99e3e3a272e73356cc0509b2c59a292))

* Removed debugging statements ([`6489d1c`](https://github.com/jbloomAus/SAEDashboard/commit/6489d1c5b52ed86cb280c237c08e10238e0d0564))

* more debug prints x3 ([`5ba2b8a`](https://github.com/jbloomAus/SAEDashboard/commit/5ba2b8a69f1881b901131976c7d52f142068dbd2))

* more debug prints x2 ([`e124ff9`](https://github.com/jbloomAus/SAEDashboard/commit/e124ff906ec7b37083af4e4721b9e33902146e47))

* more debug prints ([`e2b0c35`](https://github.com/jbloomAus/SAEDashboard/commit/e2b0c35467e5d405abd3cca664dfd1960dbba0eb))

* temp print statements ([`95df55b`](https://github.com/jbloomAus/SAEDashboard/commit/95df55b29f9250f67c5b986216e587c37f72aa9e))

* Lowered default threshold ([`dc1f31a`](https://github.com/jbloomAus/SAEDashboard/commit/dc1f31a55400231e46feb58a8c100f66472baa1b))

* updated ignore ([`eb0d56a`](https://github.com/jbloomAus/SAEDashboard/commit/eb0d56a9f813b9cf82742093fae00bb0ccfdac45))

* Reduced memory load of GQA DFA ([`05867f1`](https://github.com/jbloomAus/SAEDashboard/commit/05867f1d0c8b5f2a5b76f3ea45ab9c87eaae9c09))

* DFA will now work for models with grouped query attention ([`91a5dd1`](https://github.com/jbloomAus/SAEDashboard/commit/91a5dd17a2e567efa7d8a89d228eb7de47ae6766))

* Added head attr weights functionality for when DFA is use ([`03a615f`](https://github.com/jbloomAus/SAEDashboard/commit/03a615f7c70a6f6e634845dab4051874698fac5b))


## v0.5.0 (2024-08-25)

### Feature

* feat: accelerate caching. Torch load / save faster when files are small. 

Refactor/accelerate caching ([`6027d0a`](https://github.com/jbloomAus/SAEDashboard/commit/6027d0a3fc0d70908bad036a9658caa406d9f809))

### Unknown

* Updated formatting ([`c1ea288`](https://github.com/jbloomAus/SAEDashboard/commit/c1ea2882a17e0d1b7b28743a34fca9d0754bd8a7))

* Sped up caching with native torch functions ([`230840a`](https://github.com/jbloomAus/SAEDashboard/commit/230840aea50b8b7055a6aa61961d7ac50855b763))

* Increased cache loading speed ([`83fe5f4`](https://github.com/jbloomAus/SAEDashboard/commit/83fe5f4bdf1252d533f203bc3f53ea9f71880ab8))


## v0.4.0 (2024-08-22)

### Feature

* feat: Refactor json writer and trigger DFA release

JSON writer has been refactored for reusability and readability ([`664f487`](https://github.com/jbloomAus/SAEDashboard/commit/664f4874b585c5510d2d3dd639c5e893023f6332))

### Unknown

* Merge pull request #20 from jbloomAus/feature/dfa

SAEVisRunner DFA Implementation ([`926ea87`](https://github.com/jbloomAus/SAEDashboard/commit/926ea87dd344548489201f68cc92b33662430813))

* Refactored JSON creation from the neuronpedia runner ([`d6bb24b`](https://github.com/jbloomAus/SAEDashboard/commit/d6bb24b6d773874d8e99be4d84402d559741907b))

* Update ci.yaml ([`4b2807d`](https://github.com/jbloomAus/SAEDashboard/commit/4b2807dd865904120d236b355c0ccb1680c2919e))

* Fixed formatting ([`a62cc8f`](https://github.com/jbloomAus/SAEDashboard/commit/a62cc8f1bdd4c6e49b76d2d594e5a6b4b8183a8c))

* Fixed target index ([`ca2668d`](https://github.com/jbloomAus/SAEDashboard/commit/ca2668da03ea4d06cdc9f198988b80e0db844316))

* Corrected DFA indexing ([`d5028ae`](https://github.com/jbloomAus/SAEDashboard/commit/d5028aec875db4c03196726400c3b90b5d9d4d01))

* Adding temporary testing notebook ([`98e4b2f`](https://github.com/jbloomAus/SAEDashboard/commit/98e4b2f93d300ad4e94985d8d2594739a277e0c8))

* Added DFA output to neuronpedia runner ([`68eeff3`](https://github.com/jbloomAus/SAEDashboard/commit/68eeff3172b0c8637a6566c07951c28fd14a1c03))

* Fixed test typehints ([`d358e6f`](https://github.com/jbloomAus/SAEDashboard/commit/d358e6f5cc37304935eed949a0b0b985ba12b94f))

* Fixed formatting ([`5cb19e2`](https://github.com/jbloomAus/SAEDashboard/commit/5cb19e241051503730b6982813a6730556990c92))

* Corrected typehints ([`6173fbd`](https://github.com/jbloomAus/SAEDashboard/commit/6173fbd3824b7cba58e1cf0c7ee239762ee533ce))

* Removed another unused import ([`8be1572`](https://github.com/jbloomAus/SAEDashboard/commit/8be1572370b1adf341e2a650953bf17cd179808d))

* Removed unused imports ([`9071210`](https://github.com/jbloomAus/SAEDashboard/commit/90712105f74b287d77a06c045e8c32fd05f2e668))

* Added support for DFA calculations up to SAE Vis runner ([`4a08ffd`](https://github.com/jbloomAus/SAEDashboard/commit/4a08ffd13a8f29ff16808a20cd663c9d2d369e6a))

* Added activation collection flow for DFA ([`0ebb1f3`](https://github.com/jbloomAus/SAEDashboard/commit/0ebb1f3ca61603662f4f2cc8b1341470bf75b5d1))

* Merge pull request #19 from jbloomAus/fix/remove_precision_reduction

Removed precision reduction option ([`a5f8df1`](https://github.com/jbloomAus/SAEDashboard/commit/a5f8df15ef8619c4d08655e777d379a05b453346))

* Removed float16 option entirely from quantile calc ([`1b6a4a9`](https://github.com/jbloomAus/SAEDashboard/commit/1b6a4a93403ca2e9a869aa73600f37960090f03d))

* Removed precision reduction option ([`cd03ffb`](https://github.com/jbloomAus/SAEDashboard/commit/cd03ffb182e93a42480c01408b47ebae94d4c349))


## v0.3.0 (2024-08-15)

### Feature

* feat: seperate files per dashboard html ([`cd8d050`](https://github.com/jbloomAus/SAEDashboard/commit/cd8d050218ae3c6eeb7a9779072e60b78bfe0b58))

### Unknown

* Merge pull request #17 from jbloomAus/refactor/remove_enc_b

Removed all encoder B code ([`67c9c3f`](https://github.com/jbloomAus/SAEDashboard/commit/67c9c3fdc8bd220938f65c1f97214034cc7528b4))

* Merge pull request #18 from jbloomAus/feat-seperate-files-per-html-dashboard

feat: seperate files per dashboard html ([`8ff69ba`](https://github.com/jbloomAus/SAEDashboard/commit/8ff69ba207692d4acb8d5fc19d038090067690df))

* Removed all encoder B code ([`5174e2e`](https://github.com/jbloomAus/SAEDashboard/commit/5174e2e161030dc756c148f1740e50c52baf6a91))

* Merge pull request #16 from jbloomAus/performance_refactor

Create() will now reduce precision by default ([`fb07b90`](https://github.com/jbloomAus/SAEDashboard/commit/fb07b90eaac395a58f02ba927460dcc2c9e61d1a))

* Removed line ([`d795490`](https://github.com/jbloomAus/SAEDashboard/commit/d795490c1c9d8193c8cf84d0352b9d93c41947fe))

* Removed unnecessary print ([`4544f86`](https://github.com/jbloomAus/SAEDashboard/commit/4544f86472480f0df00344fa84111a7c2a52fcef))

* Precision will now be reduced by default for quantile calc ([`539d222`](https://github.com/jbloomAus/SAEDashboard/commit/539d222ded9e3a0944f5240f3a4cd84497d11a74))

* Merge pull request #15 from jbloomAus/quantile_efficiency

Quantile OOM prevention ([`4a40c37`](https://github.com/jbloomAus/SAEDashboard/commit/4a40c3704aab9363163fef3e2830d42f2fecdc6b))

* Made quantile batch optional and removed sampling code ([`2df51d3`](https://github.com/jbloomAus/SAEDashboard/commit/2df51d353f818a196916a15f2bc56f70480dd853))

* Added device check for test ([`afbb960`](https://github.com/jbloomAus/SAEDashboard/commit/afbb960d3c9376ad512607146826b7d1c1e68d48))

* Added parameter for quantile calculation batching ([`49d0a7a`](https://github.com/jbloomAus/SAEDashboard/commit/49d0a7ab37896a085f80409900e3d0b261b8c9e0))

* Added type annotation ([`c71c4aa`](https://github.com/jbloomAus/SAEDashboard/commit/c71c4aa1c8bc25d85b9a955b482823cbde445a51))

* Removed unused imports ([`ec01bfe`](https://github.com/jbloomAus/SAEDashboard/commit/ec01bfefc2f0f4d880cd5744ff6a2ea71991349b))

* Added float16 version of quantile calculation ([`2f01eb8`](https://github.com/jbloomAus/SAEDashboard/commit/2f01eb8d9f84a20918f19e81c23df86ddc9d7f0c))

* Merge pull request #13 from jbloomAus/hook_z_support

fix: restore hook_z support following regression. ([`ea87559`](https://github.com/jbloomAus/SAEDashboard/commit/ea87559359f9821e352dcab582e23b42fef1cebf))

* format ([`21e3617`](https://github.com/jbloomAus/SAEDashboard/commit/21e3617196ef57944c141563e9263101baf9c7f1))

* make sure hook_z works ([`efaeec0`](https://github.com/jbloomAus/SAEDashboard/commit/efaeec0fdf8c2c43bb13bfd652b812a38ebc0200))

* Merge pull request #12 from jbloomAus/use_sae_lens_loading

Use sae lens loading ([`89bba3e`](https://github.com/jbloomAus/SAEDashboard/commit/89bba3e7a10877782608c50f4b8dd9054f204381))

* add settings.json ([`d8f3034`](https://github.com/jbloomAus/SAEDashboard/commit/d8f3034c0ed7241c35e9761d60a9ee4072403fd0))

* add dtype ([`0d8008a`](https://github.com/jbloomAus/SAEDashboard/commit/0d8008afe93a2a2a5bfc954571c680a529ab883f))

* cli util ([`9da440e`](https://github.com/jbloomAus/SAEDashboard/commit/9da440eb3d50d48a7fdc4d3ee3d26de13a458593))

* wandb logging improvement ([`a077369`](https://github.com/jbloomAus/SAEDashboard/commit/a077369ca43009f4e50c0b1e7176cae398703856))

* add override for np set name ([`8906d10`](https://github.com/jbloomAus/SAEDashboard/commit/8906d103ab8d10bd01b791331dfc5485ac047a4f))

* auto add folder path to output dir ([`35e06ab`](https://github.com/jbloomAus/SAEDashboard/commit/35e06ab89bce257fc15ffaa4918b9598577d6df0))

* update tests ([`50163b0`](https://github.com/jbloomAus/SAEDashboard/commit/50163b04ca29b492b9fb71244aa26798655b663f))

* first step towards sae_lens remote loading ([`415a2d1`](https://github.com/jbloomAus/SAEDashboard/commit/415a2d1e484e9ea2351bf98de221f6a83a805107))


## v0.2.3 (2024-08-06)

### Fix

* fix: neuronpedia uses api_key for uploading features, and update sae_id-&gt;sae_set ([`0336a35`](https://github.com/jbloomAus/SAEDashboard/commit/0336a3587f825f0be15af79cc9a0033dda3d4a3f))

### Unknown

* Merge pull request #11 from jbloomAus/ignore_bos_option

Ignore bos option ([`ae34b70`](https://github.com/jbloomAus/SAEDashboard/commit/ae34b70b61993b4cce49a758bf85514410c67bd8))

* change threshold ([`4a0be67`](https://github.com/jbloomAus/SAEDashboard/commit/4a0be67622826f879191ced225c8c075d34bfe56))

* type fix ([`525b6a1`](https://github.com/jbloomAus/SAEDashboard/commit/525b6a10331b9fa0a464ae0c7f01af90ae97d0bb))

* default ignore bos eos pad ([`d2396a7`](https://github.com/jbloomAus/SAEDashboard/commit/d2396a714dd9ea3d59e516aa0fe30a9c9225e22f))

* ignore bos tokens ([`96cf6e9`](https://github.com/jbloomAus/SAEDashboard/commit/96cf6e9427cadf13fa13b55b7d1bc83ae81d9ec0))

* jump relu support in feature masking context ([`a1ba87a`](https://github.com/jbloomAus/SAEDashboard/commit/a1ba87a5c5e03687d7d7b5c5677bd9773fa49517))

* depend on latest sae lens ([`4988207`](https://github.com/jbloomAus/SAEDashboard/commit/4988207abaca24256f52235e474fe5fbb5028c1a))

* Merge pull request #10 from jbloomAus/auth_and_sae_set

fix: neuronpedia uses api_key for uploading features, and update sae_id -&gt; sae_set ([`4684aca`](https://github.com/jbloomAus/SAEDashboard/commit/4684aca54b69dbc913c1122f1a322ed4d808dce0))

* Combine upload-features and upload-dead-stubs ([`faac839`](https://github.com/jbloomAus/SAEDashboard/commit/faac8398fee8582b12c2d1a29df6d4de7e542bed))

* Activation store device should be cuda when available ([`93050b1`](https://github.com/jbloomAus/SAEDashboard/commit/93050b1f5c2b87c8e889fe3449d440016c996762))

* Activation store device should be cuda when available ([`4469066`](https://github.com/jbloomAus/SAEDashboard/commit/4469066af06bb4944832f2e596e36afa09adf160))

* Better support for huggingface dataset path ([`3dc4b78`](https://github.com/jbloomAus/SAEDashboard/commit/3dc4b783a1ced7b938ab45c4d10effedd148a829))

* Docker tweak ([`a1a70cb`](https://github.com/jbloomAus/SAEDashboard/commit/a1a70cb28c726887de9439024b7b1d01082d3932))


## v0.2.2 (2024-07-12)

### Fix

* fix: don&#39;t sample too many tokens + other fixes

fix: don&#39;t sample too many tokens ([`b2554b0`](https://github.com/jbloomAus/SAEDashboard/commit/b2554b017e75d14b38b343fc6e0c1bcc32be2359))

* fix: don&#39;t sample too many tokens ([`0cbb2ed`](https://github.com/jbloomAus/SAEDashboard/commit/0cbb2edb480b83823dc1a98dd7e5978ecdda0d81))

### Unknown

* - Don&#39;t force manual overrides for dtype - default to SAE&#39;s dtype
- Add n_prompts_in_forward_pass to neuronpedia.py
- Add n_prompts_total, n_tokens_in_prompt, and dataset to neuronpedia artifact
- Remove NPDashboardSettings for now (just save the NPRunnerConfig later)
- Fix lint error
- Consolidate minibatch_size_features/tokens to n_feats_at_a_time and n_prompts_in_fwd_pass
- Update/Fix NP acceptance test ([`b6282c8`](https://github.com/jbloomAus/SAEDashboard/commit/b6282c83e1898e356e271af0926e2271fb23f707))

* Merge pull request #7 from jbloomAus/performance-improvement

feat: performance improvement ([`f98b3dc`](https://github.com/jbloomAus/SAEDashboard/commit/f98b3dcf84c42687dfc92fa38377edd1c3f6fa30))

* delete unused snapshots ([`4210b48`](https://github.com/jbloomAus/SAEDashboard/commit/4210b48608792adc9b841ea92a64050311e66cd6))

* format ([`de57a2d`](https://github.com/jbloomAus/SAEDashboard/commit/de57a2d84564fc0eb7d5e42799c00f73c7007cf8))

* linter ([`4725ffa`](https://github.com/jbloomAus/SAEDashboard/commit/4725ffa2cbe743aa0bb615213f11105b6911f10d))

* hope flaky tests start passing ([`8ac9e8e`](https://github.com/jbloomAus/SAEDashboard/commit/8ac9e8e93127d4ab811019fc62bbe050a9a00e2c))

* np.memmap caching and more explicit hyperparams ([`9a24186`](https://github.com/jbloomAus/SAEDashboard/commit/9a24186cc1c118725c6db7dc3c77feb815cf938f))

* Move docker&#34; ([`27b1a27`](https://github.com/jbloomAus/SAEDashboard/commit/27b1a27118bcccf54576eb1891b936bd92848f3f))

* Add docker to workflow ([`a354fa4`](https://github.com/jbloomAus/SAEDashboard/commit/a354fa47cfb005dd2304b4237f9182e2408daeed))

* Dockerignore file ([`ed9fcf3`](https://github.com/jbloomAus/SAEDashboard/commit/ed9fcf3a634cd57f6517170784d56d86431e1710))

* new versions ([`f64e54d`](https://github.com/jbloomAus/SAEDashboard/commit/f64e54df5c1b643fc3acaff7f4d40d5597edf61a))

* Add tools to docker image ([`2a70f64`](https://github.com/jbloomAus/SAEDashboard/commit/2a70f64cfd4177d807a8345e64699054dd103e8d))

* Fix docker ([`3805f20`](https://github.com/jbloomAus/SAEDashboard/commit/3805f20bff622582d16fd6603bef4b77e6bada9e))

* Fix docker image ([`7f9ff2f`](https://github.com/jbloomAus/SAEDashboard/commit/7f9ff2f9b10ce08264b2153e8191eca32f9ee48a))

* Fix NP simple test, remove check for correlated neurons/features ([`355fad5`](https://github.com/jbloomAus/SAEDashboard/commit/355fad58ab2ab036a33375c02d9006db634702b9))

* Dockerfile, small batching fix ([`4df4c51`](https://github.com/jbloomAus/SAEDashboard/commit/4df4c5138341a1c233c3d0fe1a3d399846e92407))

* set sae_device, activation_store device ([`6d65b22`](https://github.com/jbloomAus/SAEDashboard/commit/6d65b22ef541326cc9558119b40baeb95cc2e47e))

* Fix NP dtype error ([`8bb4d9d`](https://github.com/jbloomAus/SAEDashboard/commit/8bb4d9de0c75ffed5daaba4d5ec563fbbee38f86))

* format ([`f667d92`](https://github.com/jbloomAus/SAEDashboard/commit/f667d92d9359e5c7976e21e821ac0dde8a081da6))

* depend on latest sae_lens ([`4a2a6a0`](https://github.com/jbloomAus/SAEDashboard/commit/4a2a6a0fd70d7b4a3f1f870a510a800b31f57264))

* use a much better method for getting subsets of feature activations ([`7101f13`](https://github.com/jbloomAus/SAEDashboard/commit/7101f13e13b4de5659623433ec359ecf2142daef))

* add to gitignore ([`20180e0`](https://github.com/jbloomAus/SAEDashboard/commit/20180e06a279ef93d6127b467511911db352bce5))

* add isort ([`3ab0fda`](https://github.com/jbloomAus/SAEDashboard/commit/3ab0fdaf75f735ec2eedc904529909111d0db0de))


## v0.2.1 (2024-07-08)

### Fix

* fix: trigger release ([`87bf0b5`](https://github.com/jbloomAus/SAEDashboard/commit/87bf0b5f21f0d1f5397e514090601ec21c718e35))

### Unknown

* Merge pull request #6 from jbloomAus/fix-bfloat16

fix bfloat 16 error ([`2f3c597`](https://github.com/jbloomAus/SAEDashboard/commit/2f3c597c1795357679e92caec3dd7e522c669fdb))

* fix bfloat 16 error ([`63c3c62`](https://github.com/jbloomAus/SAEDashboard/commit/63c3c62f0a03e5656ed78cc0e8f853bea3f0938e))

* Merge pull request #5 from jbloomAus/np-updates

Updates + fixes for Neuronpedia ([`9e6b5c4`](https://github.com/jbloomAus/SAEDashboard/commit/9e6b5c427024b8a468b0d06e4e096c2561c35d5d))

* Fix SAELens compatibility ([`139e1a2`](https://github.com/jbloomAus/SAEDashboard/commit/139e1a2f219d790c6f8faa9be34d9fbc9403dda3))

* Rename file ([`16709ad`](https://github.com/jbloomAus/SAEDashboard/commit/16709add9ee5063b3682be34eef0aea2ddf4eceb))

* Fix type ([`6b20386`](https://github.com/jbloomAus/SAEDashboard/commit/6b2038682ca41423dda3a3597bbe88120b120262))

* Make Neuronpedia outputs an object, and add a real acceptance test ([`a5db256`](https://github.com/jbloomAus/SAEDashboard/commit/a5db2560e5f90a49257124635b3fdbee117ed860))

* Np Runner: Multi-gpu defaults ([`07f7128`](https://github.com/jbloomAus/SAEDashboard/commit/07f71282681ffa801dd15f9265be349cd5745b42))

* Ensure minibatch is on correct device ([`e206546`](https://github.com/jbloomAus/SAEDashboard/commit/e2065462c445df0e0985fb6588d4c01cb39bbef5))

* NP Runner: Automatically use multi-gpu, devices ([`bf280e6`](https://github.com/jbloomAus/SAEDashboard/commit/bf280e685dc4dd2018cd41aa94a29bc853fcee18))

* Allow dtype override ([`a40077d`](https://github.com/jbloomAus/SAEDashboard/commit/a40077dac1fa2ae880fcdabe3227878ef2cfaebe))

* NP-Runner: Remove unnecessary layer of batching. ([`e2ac92b`](https://github.com/jbloomAus/SAEDashboard/commit/e2ac92b036d0192e132c8a8700a5a2f448d1983b))

* NP Runner: Allow skipping sparsity check ([`ef74d2a`](https://github.com/jbloomAus/SAEDashboard/commit/ef74d2aeea2463afe150a5e8824da5a5206cd3d0))

* Merge pull request #2 from jbloomAus/multiple-devices

feat: Multiple devices ([`535e6c9`](https://github.com/jbloomAus/SAEDashboard/commit/535e6c9689d855f82a6ddfd9f169720fe367bde3))

* format ([`7f892ad`](https://github.com/jbloomAus/SAEDashboard/commit/7f892ad0efb42025df0bcf26bdddd6fac4c2d8b1))

* NP runner takes device args seperately ([`8fc31dd`](https://github.com/jbloomAus/SAEDashboard/commit/8fc31dd6ccd59f4f35742a4e15c380673c8cb2a3))

* multi-gpu-support ([`5e24e4e`](https://github.com/jbloomAus/SAEDashboard/commit/5e24e4e6598dd7943f8d677042dcf84bc6f7a0a6))


## v0.2.0 (2024-06-10)

### Feature

* feat: experimental release 2 ([`e264f97`](https://github.com/jbloomAus/SAEDashboard/commit/e264f97d90299f6ade294db8ed03aed9cd7491ee))


## v0.1.0 (2024-06-10)

### Feature

* feat: experimental release ([`d79310a`](https://github.com/jbloomAus/SAEDashboard/commit/d79310a7b6599f7b813e214c9268d736e0cb87f0))

### Unknown

* fix pyproject.toml ([`a27c87d`](https://github.com/jbloomAus/SAEDashboard/commit/a27c87da987f043b470abce3404e305ec3f0d620))

* test deployment ([`288a2d9`](https://github.com/jbloomAus/SAEDashboard/commit/288a2d9bf797a1a2f9947b1ceac5e47edc1684ba))

* refactor np runner and add acceptance test ([`212593c`](https://github.com/jbloomAus/SAEDashboard/commit/212593c33b3aec33078a121738c0a826f705722f))

* Fix: Default context tokens length for neuronpedia runner ([`aefe95c`](https://github.com/jbloomAus/SAEDashboard/commit/aefe95cb1be4139ac45f042abdc78e0feccfb490))

* Allow custom context tokens length for Neuronpedia runner ([`d204cc8`](https://github.com/jbloomAus/SAEDashboard/commit/d204cc8fbb2ef376a1a5e00cd4f1cc5db2afb279))

* Fix: Streaming default true ([`1b91dff`](https://github.com/jbloomAus/SAEDashboard/commit/1b91dff045fdbd8c118c5f209750eca60c260f5f))

* Fix n_devices error for non-cuda ([`70b2dbd`](https://github.com/jbloomAus/SAEDashboard/commit/70b2dbdb2da51f5d78b1c2ce3210865fc259c97b))

* fix import path for ci ([`3bd4687`](https://github.com/jbloomAus/SAEDashboard/commit/3bd468727e2ab0b7d77224b7c0dad88e0727b773))

* make pyright happy, start config ([`b39ae85`](https://github.com/jbloomAus/SAEDashboard/commit/b39ae85d938a0db7c70b7dff9683f68f255dfb67))

* add black ([`236855b`](https://github.com/jbloomAus/SAEDashboard/commit/236855be1ef1464ea85b2afc6aaee963326f9257))

* fix ci ([`12818d7`](https://github.com/jbloomAus/SAEDashboard/commit/12818d7e6cd3e483258598b668805c1a9a048049))

* add pytest cov ([`aae0571`](https://github.com/jbloomAus/SAEDashboard/commit/aae057159639cd247a82fdeda9eddb98612ceec6))

* bring checks in line with sae_lens ([`7cd9679`](https://github.com/jbloomAus/SAEDashboard/commit/7cd9679cc18c64a7c8a0a07a1f12e6fc87543537))

* activation scaling factor ([`333d377`](https://github.com/jbloomAus/SAEDashboard/commit/333d3770d0d1d3c40dfeb3335dcfc46e9b7da717))

* Move Neuronpedia runner to SAEDashboard ([`4e691ea`](https://github.com/jbloomAus/SAEDashboard/commit/4e691eaad919e12b9cae6ff707eaa3cf322ea030))

* fold w_dec norm by default ([`b6c9bc7`](https://github.com/jbloomAus/SAEDashboard/commit/b6c9bc70dc419d1e32bfb5580997369215e15429))

* rename sae_vis to sae_dashboard ([`f0f5341`](https://github.com/jbloomAus/SAEDashboard/commit/f0f5341ffdf31a11884777d6ba8100cd302b9dab))

* rename feature data generator ([`e02ed0a`](https://github.com/jbloomAus/SAEDashboard/commit/e02ed0a18e92c497aea3e137cf43e9f354f8f30f))

* update demo ([`8aa9e52`](https://github.com/jbloomAus/SAEDashboard/commit/8aa9e5272f54d04b741e63aa335bfa1212a2d0f7))

* add demo ([`dd3036f`](https://github.com/jbloomAus/SAEDashboard/commit/dd3036f90e6a4ed459ec21647744d491911900ac))

* delete old demo files ([`3d86202`](https://github.com/jbloomAus/SAEDashboard/commit/3d8620204cf6acb21b5e7f9983c300341345cd88))

* remove unnecessary print statement ([`9d3d937`](https://github.com/jbloomAus/SAEDashboard/commit/9d3d937e74f5575dde68d5a21fb73ce6f826d0d4))

* set sae lens version ([`87a7691`](https://github.com/jbloomAus/SAEDashboard/commit/87a76911ff0f0d46ab421d9b5107aef27216e88b))

* update older readme ([`c5c98e5`](https://github.com/jbloomAus/SAEDashboard/commit/c5c98e53531874efab5bc16235d9c72816fa61d5))

* test ([`923da42`](https://github.com/jbloomAus/SAEDashboard/commit/923da427b56178acd99b988d6d6b51368b5d2359))

* remove sae lens dep ([`2c26d5f`](https://github.com/jbloomAus/SAEDashboard/commit/2c26d5f4c40c41f750971601968577f316e15598))

* Merge branch &#39;refactor_b&#39; ([`3154d63`](https://github.com/jbloomAus/SAEDashboard/commit/3154d636e1a9f8a30b54c17e62a842bed3f8b2a1))

* pass linting ([`0c079a1`](https://github.com/jbloomAus/SAEDashboard/commit/0c079a105b1b98e0edf2ff1a15593567c81bb103))

* format ([`6f37e2e`](https://github.com/jbloomAus/SAEDashboard/commit/6f37e2eb050a3207a2d3b9defd5d416645215c7c))

* run ci on all branches ([`faa0cc4`](https://github.com/jbloomAus/SAEDashboard/commit/faa0cc4eed4ff35f1e04656a968214c4fefbd573))

* don&#39;t use feature ablations ([`dc6e6dc`](https://github.com/jbloomAus/SAEDashboard/commit/dc6e6dc2d2affce331894d8bb61942e103182652))

* mock information in sequences to make normal sequence generation pass ([`c87b82f`](https://github.com/jbloomAus/SAEDashboard/commit/c87b82fdcc5e849d970cdc8bd1e841ec3e3e48ce))

* Remove resid ([`ff83737`](https://github.com/jbloomAus/SAEDashboard/commit/ff837373b65e60d8a9ba7c6e61f78bddc4d170f2))

* adding a test for direct_effect_feature_ablation_experiment ([`a9f3d1b`](https://github.com/jbloomAus/SAEDashboard/commit/a9f3d1b8021d8eeb60cf465934037d07583fa0b2))

* shortcut direct_effect_feature_ablation_experiment if everything is zero ([`2c68ff0`](https://github.com/jbloomAus/SAEDashboard/commit/2c68ff0c8496c58cc0732f3c51905c9c9f405393))

* fixing CI and replacing manual snapshots with syrupy snapshots ([`3b97640`](https://github.com/jbloomAus/SAEDashboard/commit/3b97640803cab3e3915202ac80c43b855c69c1cb))

* more refactor, WIP ([`81657c8`](https://github.com/jbloomAus/SAEDashboard/commit/81657c8c897a81102c0df7b29c49d526e639bb44))

* continue refactor, make data generator ([`eb1ae0f`](https://github.com/jbloomAus/SAEDashboard/commit/eb1ae0fc621407b33481c50b78b041079b08393d))

* add use of safetensors cache for repeated calculations ([`a241c32`](https://github.com/jbloomAus/SAEDashboard/commit/a241c322334340a84c2a252bc0b4a40ed2f19bc9))

* more refactor / benchmarking ([`d65ee87`](https://github.com/jbloomAus/SAEDashboard/commit/d65ee87cd191b2ed279f9f6efabb9e98bb700855))

* only run unit tests ([`5f11ddd`](https://github.com/jbloomAus/SAEDashboard/commit/5f11ddd9bc25f9c9bb7cbeba11224ba12b260ea8))

* fix lint issue ([`24daf17`](https://github.com/jbloomAus/SAEDashboard/commit/24daf17cb92534901681affbcebea314e2cf6580))

* format ([`83e89ed`](https://github.com/jbloomAus/SAEDashboard/commit/83e89ed4860d886ccf19be591bf72d0e029e7344))

* organise tests, make sure only unit tests run on CI ([`21f5fb1`](https://github.com/jbloomAus/SAEDashboard/commit/21f5fb155665329531b16d10673ddd988e7034ea))

*  see if we can do some caching ([`c1dca6f`](https://github.com/jbloomAus/SAEDashboard/commit/c1dca6faa61de0849453acc83ae23baab6cf48be))

* more refactoring ([`b3f0f41`](https://github.com/jbloomAus/SAEDashboard/commit/b3f0f41f36f0eee57a08142880d4b6654309e62c))

* further refactor, possible significant speed up ([`ddd3496`](https://github.com/jbloomAus/SAEDashboard/commit/ddd3496206c0f3e751b596ca51e3544c77ddaf94))

* more refactor ([`a5f6deb`](https://github.com/jbloomAus/SAEDashboard/commit/a5f6deb4263c58803e7af23d767fc5cb17dfd2b2))

* refactoring in progress ([`d210b60`](https://github.com/jbloomAus/SAEDashboard/commit/d210b6056aa5316d2fd917e24ca8a819331a8114))

* use named arguments ([`4a81053`](https://github.com/jbloomAus/SAEDashboard/commit/4a8105355d3b86e460f32cd5c736dde0dbeaa2e3))

* remove create method ([`43b2018`](https://github.com/jbloomAus/SAEDashboard/commit/43b20184ed5ed0c2f08cfd13423f2271fd871274))

* move chunk ([`0f26aa8`](https://github.com/jbloomAus/SAEDashboard/commit/0f26aa85bc9fbe358f4c5f90971d51b86159f095))

* use fixtures ([`7c11dd9`](https://github.com/jbloomAus/SAEDashboard/commit/7c11dd914d467957e5c00b914f302c291924e411))

* refactor to create runner ([`9202c19`](https://github.com/jbloomAus/SAEDashboard/commit/9202c19f4ad6134eb6b68f857c9e4bfd0b911cf8))

* format ([`abd8747`](https://github.com/jbloomAus/SAEDashboard/commit/abd87472b76cfc151abfe2a6e312ea43b29c2250))

* target ci at this branch ([`ea3b2a3`](https://github.com/jbloomAus/SAEDashboard/commit/ea3b2a3181f2eb1ff52d83b2040b586d6fdfef4a))

* comment out release process for now ([`7084b5b`](https://github.com/jbloomAus/SAEDashboard/commit/7084b5ba3325bb559a8377d379bd2f3ba6d68348))

* test generated output ([`7b8b2ab`](https://github.com/jbloomAus/SAEDashboard/commit/7b8b2abd94213d67c378b7746107f6a7c811d93c))

* commit current demo html ([`00a03a0`](https://github.com/jbloomAus/SAEDashboard/commit/00a03a02fbf181caa55704defac25578b4444452))


## v0.0.1 (2024-04-25)

### Chore

* chore: setting up pytest ([`2079d00`](https://github.com/jbloomAus/SAEDashboard/commit/2079d00911d1a00ee19cde478b5cab61ca9c0495))

* chore: setting up semantic-release ([`09075af`](https://github.com/jbloomAus/SAEDashboard/commit/09075afbec279fb89d157f73e9a0ed47ba66d3c8))

### Fix

* fix: remove circular dep with sae lens ([`1dd9f6c`](https://github.com/jbloomAus/SAEDashboard/commit/1dd9f6cd22f879e8d6904ba72f3e52b4344433cd))

### Unknown

* Merge pull request #44 from chanind/pytest-setup

chore: setting up pytest ([`034eefa`](https://github.com/jbloomAus/SAEDashboard/commit/034eefa5a4163e9a560b574e2e255cd06f8f49a1))

* Merge pull request #43 from callummcdougall/move_saelens_dep

Remove dependency on saelens from pyproject, add to demo.ipynb ([`147d87e`](https://github.com/jbloomAus/SAEDashboard/commit/147d87ee9534d30e764851cbe73aadb5783d2515))

* Add missing matplotlib ([`572a3cc`](https://github.com/jbloomAus/SAEDashboard/commit/572a3cc79709a14117bbeafb871a33f0107600d8))

* Remove dependency on saelens from pyproject, add to demo.ipynb ([`1e6f3cf`](https://github.com/jbloomAus/SAEDashboard/commit/1e6f3cf9b2bcfb381a73d9333581c430faa531fd))

* Merge branch &#39;main&#39; of https://github.com/callummcdougall/sae_vis ([`4e7a24c`](https://github.com/jbloomAus/SAEDashboard/commit/4e7a24c37444f11d718035eede68ac728d949a20))

* fix conflicts ([`ea3d624`](https://github.com/jbloomAus/SAEDashboard/commit/ea3d624013b9aa7cbd2d6eaa7212a1f7c4ee8e28))

* Merge pull request #41 from callummcdougall/allow_disable_buffer

oops I forgot to switch back to main before pushing ([`1312cd0`](https://github.com/jbloomAus/SAEDashboard/commit/1312cd09d6e274b1163e79d2ac01f2df54c65157))

* Merge branch &#39;main&#39; into allow_disable_buffer ([`e7edf5a`](https://github.com/jbloomAus/SAEDashboard/commit/e7edf5a9bae4714bf4983ce6a19a0fe6fdf1f118))

* 16 ([`64e7018`](https://github.com/jbloomAus/SAEDashboard/commit/64e701849570d9e172dc065812c9a3e7149a9176))

* Merge pull request #40 from chanind/semantic-release-autodeploy

chore: setting up semantic-release for auto-deploy ([`a4d44d1`](https://github.com/jbloomAus/SAEDashboard/commit/a4d44d1a0e86055fb82ef41f51f0adbb7868df3c))

* version 0.2.16 ([`afca0be`](https://github.com/jbloomAus/SAEDashboard/commit/afca0be8826e0c007b5730fa9fa18454699d16a3))

* Merge pull request #38 from chanind/type-checking

Enabling type checking with Pyright ([`f1fd792`](https://github.com/jbloomAus/SAEDashboard/commit/f1fd7926f46f00dca46024377f53aa8f2db98773))

* Merge pull request #39 from callummcdougall/fix_loading_saelens_sae

FIX: SAELens new format has &#34;scaling_factor&#34; key, which causes assert to fail ([`983aee5`](https://github.com/jbloomAus/SAEDashboard/commit/983aee562aea31e90657caf8c6ab6e450e952120))

* Fix Formatting ([`13b8106`](https://github.com/jbloomAus/SAEDashboard/commit/13b81062485f5dce2568e7832bfb2aae218dd4e9))

* Merge branch &#39;main&#39; into fix_loading_saelens_sae ([`21b0086`](https://github.com/jbloomAus/SAEDashboard/commit/21b0086b8af3603441795e925a15e7cded122acb))

* Allow SAELens autoencoder keys to be superset of required keys, instead of exact match ([`6852170`](https://github.com/jbloomAus/SAEDashboard/commit/6852170d55e7d3cf22632c5807cfab219516da98))

* enabling type checking with Pyright ([`05d14ea`](https://github.com/jbloomAus/SAEDashboard/commit/05d14eafea707d3db81e78b4be87199087cb8e37))

* Fix version ([`5a43916`](https://github.com/jbloomAus/SAEDashboard/commit/5a43916cbd9836396f051f7a258fdca8664e05e9))

* format ([`8f1506b`](https://github.com/jbloomAus/SAEDashboard/commit/8f1506b6eb7dc0a2d4437d2aa23a0898c46a156d))

* v0.2.17 ([`2bb14da`](https://github.com/jbloomAus/SAEDashboard/commit/2bb14daa88a0af601e13f4e51b50a2b00cd75b48))

* Use main branch of SAELens ([`2b34505`](https://github.com/jbloomAus/SAEDashboard/commit/2b345052bdc92ee9c1255cab0978916307a0a9dc))

* Update version 0.2.16 ([`bf90293`](https://github.com/jbloomAus/SAEDashboard/commit/bf902930844db9b0f8db4fbe8b3610557352660b))

* Merge pull request #36 from callummcdougall/allow_disable_buffer

FEATURE: Allow setting buffer to None, which gives the whole activation sequence ([`f5f9594`](https://github.com/jbloomAus/SAEDashboard/commit/f5f9594fcaf5edb6036a85446e092278004ea200))

* fix all indices view ([`5f87d52`](https://github.com/jbloomAus/SAEDashboard/commit/5f87d52154d6a8e8c8984836bbe8f85ee25f279d))

* Merge pull request #35 from callummcdougall/fix_gpt2_demo

Fix usage of SAELens and demo notebook ([`88b5933`](https://github.com/jbloomAus/SAEDashboard/commit/88b59338d3cadbd5c70f0c1117dff00f01a54e6a))

* Merge branch &#39;fix_gpt2_demo&#39; into allow_disable_buffer ([`ea57bfc`](https://github.com/jbloomAus/SAEDashboard/commit/ea57bfc2ee1e23666810982abf32e6e9cbb74193))

* Import updated SAELens, use correct tokens, fix missing file cfg.json file error. ([`14ba9b0`](https://github.com/jbloomAus/SAEDashboard/commit/14ba9b03d4ce791ba8f4cac553fb82a93c47dfb8))

* Merge pull request #34 from ArthurConmy/patch-1

Update README.md ([`3faac82`](https://github.com/jbloomAus/SAEDashboard/commit/3faac82686f546800492d8aeb5e1d5919cbf1517))

* Update README.md ([`416eca8`](https://github.com/jbloomAus/SAEDashboard/commit/416eca8073c6cb2b120c759330ec47f52ab32d1e))

* Merge pull request #33 from chanind/setup-poetry-and-ruff

Setting up poetry / ruff / github actions ([`287f30f`](https://github.com/jbloomAus/SAEDashboard/commit/287f30f1d8fc39ab583f202c9277e07e5eeeaf62))

* setting up poetry and ruff for linting/formatting ([`0e0eba9`](https://github.com/jbloomAus/SAEDashboard/commit/0e0eba9e4d54c746cddc835ef4f6ddf2bab96844))

* fix feature vis demo gpt ([`821781e`](https://github.com/jbloomAus/SAEDashboard/commit/821781e96b732a5909d8735714482c965891b2ea))

* Allow disabling the buffer ([`c1be9f8`](https://github.com/jbloomAus/SAEDashboard/commit/c1be9f8e4b8ee6d8f18c4a1a0445840304440c1d))

* add scatter plot support ([`6eab28b`](https://github.com/jbloomAus/SAEDashboard/commit/6eab28bef9ef5cd9360fef73e02763301fa1a028))

* update setup ([`8d2ca53`](https://github.com/jbloomAus/SAEDashboard/commit/8d2ca53e8a6bba860fe71368741d06a718adaa27))

* fix setup ([`9cae8f4`](https://github.com/jbloomAus/SAEDashboard/commit/9cae8f461bd780e23eb2d994f56b495ede16201a))

* Merge branch &#39;main&#39; of https://github.com/callummcdougall/sae_vis ([`ed8f8cb`](https://github.com/jbloomAus/SAEDashboard/commit/ed8f8cb7ad1fba2383dcdd471c33ce4a1b9f32e3))

* fix sae bug ([`247d14b`](https://github.com/jbloomAus/SAEDashboard/commit/247d14b55f209ed9ccf50e5ce091ed66ffbf19d2))

* Merge pull request #27 from wllgrnt/will-add-eindex-dependency

Update setup.py with eindex dependency ([`8d7ed12`](https://github.com/jbloomAus/SAEDashboard/commit/8d7ed123505ac7ecf93dd310f57888547aead1d7))

* Merge pull request #32 from hijohnnylin/pin_older_sae_training

Demo notebook errors under &#34;Multi-layer models&#34; vis ([`9ac1dac`](https://github.com/jbloomAus/SAEDashboard/commit/9ac1dac51af32909666977cb5b3794965c70f62f))

* Pin older commit of mats_sae_training ([`8ca7ac1`](https://github.com/jbloomAus/SAEDashboard/commit/8ca7ac14b919fedb91240630ac7072cac40a6d6a))

* two more deps ([`7f231a8`](https://github.com/jbloomAus/SAEDashboard/commit/7f231a83acfef2494c1866249f57e10c21a1a443))

* Update setup.py with eindex

Without this, &#39;pip install sae-vis&#39; will cause errors when e.g. you do &#39;from sae_vis.data_fetching_fns import get_feature_data&#39; ([`a9d7de9`](https://github.com/jbloomAus/SAEDashboard/commit/a9d7de90b492f7305758e15303ba890fb9b503d0))

* update version number ([`72e584b`](https://github.com/jbloomAus/SAEDashboard/commit/72e584b6492ed1ef3989968f6588a17fca758650))

* add gifs to readme ([`1393740`](https://github.com/jbloomAus/SAEDashboard/commit/13937405da31cca70cd1027aaca6c9cc84797ff1))

* test gif ([`4fbafa6`](https://github.com/jbloomAus/SAEDashboard/commit/4fbafa69343dc58dc18d0f78e393b5fcc9e24c0c))

* fix height issue ([`3f272f6`](https://github.com/jbloomAus/SAEDashboard/commit/3f272f61a954effef7bd648cc8117346da3bb971))

* fix pypi ([`7151164`](https://github.com/jbloomAus/SAEDashboard/commit/7151164cc0df8af278617147f07cbfbe3977cfeb))

* update setup ([`8c43478`](https://github.com/jbloomAus/SAEDashboard/commit/8c43478ad2eba8d3d4106fe4239c1229b8720fe6))

* Merge pull request #26 from hijohnnylin/update_html_anomalies

Update and add some HTML_ANOMALIES ([`1874a47`](https://github.com/jbloomAus/SAEDashboard/commit/1874a47a099ce32795bdbb5f98b9167dcca85ff2))

* Update and add some HTML_ANOMALIES ([`c541b7f`](https://github.com/jbloomAus/SAEDashboard/commit/c541b7f06108046ad1e2eb82c89f30f061f4411e))

* 0.2.9 ([`a5c8a6d`](https://github.com/jbloomAus/SAEDashboard/commit/a5c8a6d2008b818db90566cba50211845c753444))

* fix readme ([`5a8a7e3`](https://github.com/jbloomAus/SAEDashboard/commit/5a8a7e3173fc50fdb5ff0e56d7fa83e475af38a3))

* include feature tables ([`7c4c263`](https://github.com/jbloomAus/SAEDashboard/commit/7c4c263a2e069482d341b6265015664792bde817))

* add license ([`fa02a3d`](https://github.com/jbloomAus/SAEDashboard/commit/fa02a3dc93b721322b3902e2ac416ed156bf9d80))

* Merge branch &#39;main&#39; of https://github.com/callummcdougall/sae_vis ([`ca5efcd`](https://github.com/jbloomAus/SAEDashboard/commit/ca5efcdc81074d3c3002bd997b35e326a44a4a25))

* re-fix html anomalies ([`2fbae4c`](https://github.com/jbloomAus/SAEDashboard/commit/2fbae4c9a7dd663737bae25e73e978d40c59064a))

* Merge pull request #24 from chanind/fix-pypi-repo-link

fixing repo URL in setup.py ([`14a0be5`](https://github.com/jbloomAus/SAEDashboard/commit/14a0be54a57b1bc73ac4741611f9c8d1bd229e6f))

* fixing repo URL in setup.py ([`4faeca5`](https://github.com/jbloomAus/SAEDashboard/commit/4faeca5da06c0bb4384e202a91d895a217365d30))

* fix hook point bug ([`9b573b2`](https://github.com/jbloomAus/SAEDashboard/commit/9b573b27590db1cbd6c8ef08fca7ff8c9d26b340))

* Merge pull request #20 from chanind/fix-final-resid-layer

fixing bug if hook_point == hook_point_resid_final ([`d6882e3`](https://github.com/jbloomAus/SAEDashboard/commit/d6882e3f813ef0d399e07548871f61b1f6a98ac6))

* fixing bug using hook_point_resid_final ([`cfe9b30`](https://github.com/jbloomAus/SAEDashboard/commit/cfe9b3042cfe127d5f7958064ffe817c25a19b56))

* fix indexing speed ([`865ff64`](https://github.com/jbloomAus/SAEDashboard/commit/865ff64329538641cd863dc7668dfc77907fb384))

* enable JSON saving ([`feea47a`](https://github.com/jbloomAus/SAEDashboard/commit/feea47a342d52296b72784ed18ea628848d4c7d4))

* Merge pull request #19 from chanind/support-mlp-and-attn-out

supporting mlp and attn out hooks ([`1c5463b`](https://github.com/jbloomAus/SAEDashboard/commit/1c5463b12f85cd0598b4e2fba5c556b1e9c0fbbe))

* supporting mlp and attn out hooks ([`a100e58`](https://github.com/jbloomAus/SAEDashboard/commit/a100e586498e8cae14df475bc7924cdecaed71ea))

* Merge branch &#39;main&#39; of https://github.com/callummcdougall/sae_vis ([`083aeba`](https://github.com/jbloomAus/SAEDashboard/commit/083aeba0e4048d9976ec5cbee8df7dc8fd4db4e9))

* fix variable naming ([`2507918`](https://github.com/jbloomAus/SAEDashboard/commit/25079186b3f31d2271b1ecdb11f26904af7146d2))

* Merge pull request #18 from chanind/remove-build-artifacts

removing Python build artifacts and adding to .gitignore ([`b0e0594`](https://github.com/jbloomAus/SAEDashboard/commit/b0e0594590b4472b34052c6eb3ebceb6c9f58a11))

* removing Python build artifacts and adding to .gitignore ([`b6486f5`](https://github.com/jbloomAus/SAEDashboard/commit/b6486f56bea9d4bb7544c36afe70e6f891101b63))

* update readme ([`0ee3608`](https://github.com/jbloomAus/SAEDashboard/commit/0ee3608af396a1a6586dfb809f2f6480bb4f6390))

* update readme ([`f8351f8`](https://github.com/jbloomAus/SAEDashboard/commit/f8351f88e8432ccd4b2206e859daea316304d6c6))

* update version number ([`1e74408`](https://github.com/jbloomAus/SAEDashboard/commit/1e7440883f44a92705299430215f802fea4e1915))

* fix formatting and docstrings ([`b9fe2bb`](https://github.com/jbloomAus/SAEDashboard/commit/b9fe2bbb15a48e4b0415f6f4240d895990d54c9a))

* Merge pull request #17 from jordansauce/sae-agnostic-functions-new

Added SAE class agnostic functions ([`0039c6f`](https://github.com/jbloomAus/SAEDashboard/commit/0039c6f8f99d6e8a1b2ff56aa85f60a3eba3afb0))

* add to pypi ([`02a5b9a`](https://github.com/jbloomAus/SAEDashboard/commit/02a5b9acd15433cc59d438271b9bd5e12d62b662))

* Added sae class agnostic functions

Added parse_feature_data() and parse_prompt_data() ([`e2709d0`](https://github.com/jbloomAus/SAEDashboard/commit/e2709d0b4c55d73d6026f3b9ce534f59ce61f344))

* update notebook images ([`b87ad4d`](https://github.com/jbloomAus/SAEDashboard/commit/b87ad4d256f12c23605b0e7db307ee56913c93ef))

* fix layer parse and custom device ([`14c7ae9`](https://github.com/jbloomAus/SAEDashboard/commit/14c7ae9d0c8b7dad21b953cfc93fe7f34c74e149))

* update dropdown styling ([`83be219`](https://github.com/jbloomAus/SAEDashboard/commit/83be219bfe31b985a26762e06345c574aa0e6fe1))

* add custom prompt vis ([`cabdc5c`](https://github.com/jbloomAus/SAEDashboard/commit/cabdc5cb31f881cddf236490c41332c525d2ee74))

* d3 &amp; multifeature refactor ([`f79a919`](https://github.com/jbloomAus/SAEDashboard/commit/f79a919691862f60a9e30fe0f79fd8e771bc932a))

* remove readme links ([`4bcef48`](https://github.com/jbloomAus/SAEDashboard/commit/4bcef489b644dd3357b1975f3245d534f6f0d2e0))

* add demo html ([`629c713`](https://github.com/jbloomAus/SAEDashboard/commit/629c713345407562dc4ccd9875bf3cfab5480bdd))

* remove demos ([`beedea9`](https://github.com/jbloomAus/SAEDashboard/commit/beedea9667761534a5293015aff9cc17638666a5))

* fix quantile error ([`3a23cfd`](https://github.com/jbloomAus/SAEDashboard/commit/3a23cfd56f21fe0775a1a9957db340d15f75f51a))

* width 425 ([`f25c776`](https://github.com/jbloomAus/SAEDashboard/commit/f25c776d5cb746916d3f2fdf368cbd5448742949))

* fix device bug ([`85dfa49`](https://github.com/jbloomAus/SAEDashboard/commit/85dfa497bc804945911e80607ac31cf3afbdc759))

* dont return vocab dict ([`b4c7138`](https://github.com/jbloomAus/SAEDashboard/commit/b4c713873870acb4035986cc5bff3a4ce1e466c9))

* save as JSON, fix device ([`eba2cff`](https://github.com/jbloomAus/SAEDashboard/commit/eba2cff3eb6215558577a6b4d4f8cc716766b927))

* simple fixed and issues ([`b28a0f7`](https://github.com/jbloomAus/SAEDashboard/commit/b28a0f7c7e936f4bea05528d952dfcd438533cce))

* Merge pull request #8 from lucyfarnik/topk-empty-mask

Topk error handling for empty masks ([`2740c00`](https://github.com/jbloomAus/SAEDashboard/commit/2740c0047e78df7e56d7bcf707c909ac18e71c1f))

* Topk error handling for empty masks ([`1c2627e`](https://github.com/jbloomAus/SAEDashboard/commit/1c2627e237f8f67725fc44e60a190bc141d36fc8))

* viz to vis ([`216d02b`](https://github.com/jbloomAus/SAEDashboard/commit/216d02b550d6fbcb9b37d39c1b272a7dda91aadc))

* update readme links ([`f9b3f95`](https://github.com/jbloomAus/SAEDashboard/commit/f9b3f95e31e7150024be27ec62246f43bf9bcbb8))

* update for TL ([`1941db1`](https://github.com/jbloomAus/SAEDashboard/commit/1941db1e22093d6fc88fb3fcd6f4c7d535d8b3b4))

* Merge pull request #5 from lucyfarnik/transformer-lens-models

Compatibility with TransformerLens models ([`8d59c6c`](https://github.com/jbloomAus/SAEDashboard/commit/8d59c6c5a5f2b98c486e5c74130371ad9254d1c9))

* Merge branch &#39;main&#39; into transformer-lens-models ([`73057d7`](https://github.com/jbloomAus/SAEDashboard/commit/73057d7e2a3e4e9669fc0556e64190811ac8b52d))

* Merge pull request #4 from lucyfarnik/resid-saes-support

Added support for residual-adjacent SAEs ([`b02e98b`](https://github.com/jbloomAus/SAEDashboard/commit/b02e98b3b852c0613a890f8949d04b5560fb6fd6))

* Merge pull request #7 from lucyfarnik/fix-histogram-div-zero

Fixed division by zero in histogram calculation ([`3aee20e`](https://github.com/jbloomAus/SAEDashboard/commit/3aee20ea7f99cc07e6c5085fddb70cadd8327f4d))

* Merge pull request #6 from lucyfarnik/handling-dead-features

Edge case handling for dead features ([`9e43c30`](https://github.com/jbloomAus/SAEDashboard/commit/9e43c308e58769828234e1505f1c1102ba651dfd))

* add features argument ([`f24ef7e`](https://github.com/jbloomAus/SAEDashboard/commit/f24ef7ebebb3d4fd92e299858dbd5b968b78c69e))

* fix image link ([`22c8734`](https://github.com/jbloomAus/SAEDashboard/commit/22c873434dfa84e3aed5ee0aab0fd25b288428a6))

* Merge pull request #1 from lucyfarnik/read-me-links-fix

Fixed readme links pointing to the old colab ([`86f8e20`](https://github.com/jbloomAus/SAEDashboard/commit/86f8e2012e376b6c498e5e708324f812af6fbc98))

* Fixed division by zero in histogram calculation ([`e986e90`](https://github.com/jbloomAus/SAEDashboard/commit/e986e907cc42790efc93ce75ebf7b28a0278aaa2))

* Added readme section about models ([`7523e7f`](https://github.com/jbloomAus/SAEDashboard/commit/7523e7f6363e030196496b3c6a3dc70b234c2d9a))

* Fixed readme links pointing to the old colab ([`28ef1cb`](https://github.com/jbloomAus/SAEDashboard/commit/28ef1cbd1b91f6c09c842f48e1f997d189ca04e7))

* Edge case handling for dead features ([`5197aee`](https://github.com/jbloomAus/SAEDashboard/commit/5197aee2c9f92bce7c5fd6d22201152a68c2e6ca))

* Compatibility with TransformerLens models ([`ba708e9`](https://github.com/jbloomAus/SAEDashboard/commit/ba708e987be6cc7a09d34ea8fb83de009312684d))

* Added support for MPS ([`196c0a2`](https://github.com/jbloomAus/SAEDashboard/commit/196c0a24d0e8277b327eb2d57662075f9106990b))

* Added support for residual-adjacent SAEs ([`89aacf1`](https://github.com/jbloomAus/SAEDashboard/commit/89aacf1b22aa81b393b10eca8611c9dbf406c638))

* black font ([`d81e74d`](https://github.com/jbloomAus/SAEDashboard/commit/d81e74d575326ef786881fb9182a768f9de2cb70))

* fix html bug ([`265dedd`](https://github.com/jbloomAus/SAEDashboard/commit/265dedd376991230e2041fd37d5b6a0eda048545))

* add jax and dataset deps ([`f1caeaf`](https://github.com/jbloomAus/SAEDashboard/commit/f1caeafc9613e27c7663447cf862301ac11d842d))

* remove TL dependency ([`155991f`](https://github.com/jbloomAus/SAEDashboard/commit/155991fe61d0199d081d344ac44996edce35d118))

* first commit ([`7782eb6`](https://github.com/jbloomAus/SAEDashboard/commit/7782eb6d5058372630c5bbb8693eb540a7bceaf4))
