package decode

import "encoding/binary"

// getScaleMinK4 从 Q4_K / Q5_K 的 12 字节 scales 里取第 j 组（共 8 组）的
// 6 位 scale 与 6 位 min。
//
// 布局不直观：j < 4 时低 4 位在前 4 字节、高 4 位在后 4 字节；
// j >= 4 时低 4 位取 bytes[j+4] 的低半字节，高 2 位分别从 bytes[j-4]
// 与 bytes[j] 的高 2 位拼出来。
func getScaleMinK4(j int, q []byte) (sc, m uint8) {
	if j < 4 {
		return q[j] & 63, q[j+4] & 63
	}
	return (q[j+4] & 0x0F) | ((q[j-4] >> 6) << 4),
		(q[j+4] >> 4) | ((q[j] >> 6) << 4)
}

// b2u16 把"某位是否置位"变成 0/1，用于拼出第 5 个比特。
func b2u16(v uint8) uint16 {
	if v != 0 {
		return 1
	}
	return 0
}

// Q4_K：144 字节 = d(2) + dmin(2) + scales(12) + qs(128)，256 个权重分 8 个 32 元素的子块。
//
// 每个子块有自己的 scale 与 min（由 getScaleMinK4 从 6 位字段还原），
// 实际系数是 d*sc 与 dmin*m —— 分两级缩放，这是 K 系列与 Q4_0 的根本区别。
func dqQ4K(src []byte, dst []float32) error {
	const perBlock, elems = 144, 256
	for b := 0; b*elems < len(dst); b++ {
		blk := src[b*perBlock:]
		d := f16ToF32(binary.LittleEndian.Uint16(blk))
		dmin := f16ToF32(binary.LittleEndian.Uint16(blk[2:]))
		scales := blk[4:16]
		q := blk[16:144]
		out := dst[b*elems : (b+1)*elems]

		pos, is := 0, 0
		for range 4 {
			sc1, m1 := getScaleMinK4(is, scales)
			sc2, m2 := getScaleMinK4(is+1, scales)
			d1, mm1 := d*float32(sc1), dmin*float32(m1)
			d2, mm2 := d*float32(sc2), dmin*float32(m2)
			for l := range 32 {
				out[pos] = d1*float32(q[l]&0x0F) - mm1
				pos++
			}
			for l := range 32 {
				out[pos] = d2*float32(q[l]>>4) - mm2
				pos++
			}
			q = q[32:]
			is += 2
		}
	}
	return nil
}

// Q5_K：176 字节 = d(2) + dmin(2) + scales(12) + qh(32) + qs(128)。
//
// qh 每字节提供 8 个元素的第 5 位，但**不是**按位顺序一对一：
// 第 l 个元素用的是 qh[l] 的最低两位之一（掩码 u1 依次为 1,4,16,64），
// 高半字节的元素用同一字节的次低位（u2 依次为 2,8,32,128）。
// 写错时同样是"一半对一半错"。
func dqQ5K(src []byte, dst []float32) error {
	const perBlock, elems = 176, 256
	for b := 0; b*elems < len(dst); b++ {
		blk := src[b*perBlock:]
		d := f16ToF32(binary.LittleEndian.Uint16(blk))
		dmin := f16ToF32(binary.LittleEndian.Uint16(blk[2:]))
		scales, qh, qs := blk[4:16], blk[16:48], blk[48:176]
		out := dst[b*elems : (b+1)*elems]

		pos, is := 0, 0
		u1, u2 := uint8(1), uint8(2)
		for range 4 {
			sc1, m1 := getScaleMinK4(is, scales)
			sc2, m2 := getScaleMinK4(is+1, scales)
			d1, mm1 := d*float32(sc1), dmin*float32(m1)
			d2, mm2 := d*float32(sc2), dmin*float32(m2)
			for l := range 32 {
				out[pos] = d1*float32(uint16(qs[l]&0x0F)+b2u16(qh[l]&u1)*16) - mm1
				pos++
			}
			for l := range 32 {
				out[pos] = d2*float32(uint16(qs[l]>>4)+b2u16(qh[l]&u2)*16) - mm2
				pos++
			}
			qs = qs[32:]
			u1 <<= 2
			u2 <<= 2
			is += 2
		}
	}
	return nil
}

// Q6_K：210 字节 = ql(128) + qh(64) + scales(16 个 int8) + d(2)。
//
// **d 在块尾（偏移 208），不在块头** —— 这是 K 系列里唯一一个这样的。
// 按惯例从块头取 d 的话，读到的会是 ql 的头两个字节，值会大得离谱但不报错。
func dqQ6K(src []byte, dst []float32) error {
	const perBlock, elems = 210, 256
	for b := 0; b*elems < len(dst); b++ {
		blk := src[b*perBlock:]
		d := f16ToF32(binary.LittleEndian.Uint16(blk[208:]))
		ql, qh, sc := blk[0:128], blk[128:192], blk[192:208]
		out := dst[b*elems : (b+1)*elems]

		pos := 0
		for range 2 {
			for l := range 32 {
				is := l / 16
				q1 := int8((ql[l]&0x0F)|(((qh[l]>>0)&3)<<4)) - 32
				q2 := int8((ql[l+32]&0x0F)|(((qh[l]>>2)&3)<<4)) - 32
				q3 := int8((ql[l]>>4)|(((qh[l]>>4)&3)<<4)) - 32
				q4 := int8((ql[l+32]>>4)|(((qh[l]>>6)&3)<<4)) - 32
				out[pos+l] = d * float32(int8(sc[is])) * float32(q1)
				out[pos+l+32] = d * float32(int8(sc[is+2])) * float32(q2)
				out[pos+l+64] = d * float32(int8(sc[is+4])) * float32(q3)
				out[pos+l+96] = d * float32(int8(sc[is+6])) * float32(q4)
			}
			pos += 128
			ql, qh, sc = ql[64:], qh[32:], sc[8:]
		}
	}
	return nil
}

// Q2_K：84 字节 = scales(16) + qs(64) + d(2) + dmin(2)，256 个权重。
//
// 分组顺序**不是**"逐 16 字节走完 4 个 shift"。每 32 字节是一个 chunk，
// chunk 内对 4 个 shift 各处理"前 16 字节 + 后 16 字节"两个组，
// 且这两组各用一个**独立的** scale 字节（共 2 chunk × 4 shift × 2 组 = 16 组）。
//
// 两半共用 scale 的写法会让前 16 个值对、从第 17 个起全错。
func dqQ2K(src []byte, dst []float32) error {
	const perBlock, elems = 84, 256
	for b := 0; b*elems < len(dst); b++ {
		blk := src[b*perBlock:]
		scales, qs := blk[0:16], blk[16:80]
		d := f16ToF32(binary.LittleEndian.Uint16(blk[80:]))
		dmin := f16ToF32(binary.LittleEndian.Uint16(blk[82:]))
		out := dst[b*elems : (b+1)*elems]

		pos, is := 0, 0
		for c := range 2 {
			q := qs[c*32 : (c+1)*32]
			for shift := range 4 {
				sc := scales[is]
				is++
				dl := d * float32(sc&0x0F)
				ml := dmin * float32(sc>>4)
				for l := range 16 {
					out[pos] = dl*float32((q[l]>>(2*shift))&3) - ml
					pos++
				}
				// 后半 16 字节用自己的 scale 字节，不是共用上面那个
				sc = scales[is]
				is++
				dl = d * float32(sc&0x0F)
				ml = dmin * float32(sc>>4)
				for l := range 16 {
					out[pos] = dl*float32((q[16+l]>>(2*shift))&3) - ml
					pos++
				}
			}
		}
	}
	return nil
}

// qhOffset 返回 Q3_K 的高位修正：位**置位**时偏移为 0，清零时为 4。
func qhOffset(hmByte, m uint8) int8 {
	if hmByte&m != 0 {
		return 0
	}
	return 4
}

// Q3_K：110 字节 = hmask(32) + qs(64) + scales(12) + d(2)。
//
// 三处反直觉，全部靠与参考实现逐值比对才确定：
//   - 6 位 scale 被打散：低 4 位在 scales[0..7]（每字节两个），
//     高 2 位在 scales[8..11]（每字节四个）
//   - hmask 那一位**置位时偏移是 0，清零时才是 -4**（与直觉相反）
//   - m 跨外层迭代累积：0..3 位给前 128 个元素，4..7 位给后 128 个，不重置
func dqQ3K(src []byte, dst []float32) error {
	const perBlock, elems = 110, 256
	for b := 0; b*elems < len(dst); b++ {
		blk := src[b*perBlock:]
		hmask, qs, raw := blk[0:32], blk[32:96], blk[96:108]
		dAll := f16ToF32(binary.LittleEndian.Uint16(blk[108:]))
		out := dst[b*elems : (b+1)*elems]

		// 6 位 scale 解包。按原生 32 位字做原地重排，与参考实现同构。
		//
		// 注意三个赋值用了原始的 aux[0]/aux[1] 的高 4 位，
		// 而 aux[0]/aux[1] 最后才被覆盖 —— 顺序不能调换。
		var aux [4]uint32
		for i := range 3 {
			aux[i] = binary.LittleEndian.Uint32(raw[i*4:])
		}
		const mask1 uint32 = 0x03030303
		const mask2 uint32 = 0x0f0f0f0f
		tmp := aux[2]
		aux[2] = ((aux[0] >> 4) & mask2) | (((tmp >> 4) & mask1) << 4)
		aux[3] = ((aux[1] >> 4) & mask2) | (((tmp >> 6) & mask1) << 4)
		aux[0] = (aux[0] & mask2) | (((tmp >> 0) & mask1) << 4)
		aux[1] = (aux[1] & mask2) | (((tmp >> 2) & mask1) << 4)

		var sc [16]int8
		for i := range 4 {
			for k := range 4 {
				sc[i*4+k] = int8(byte(aux[i]>>(8*k))) - 32
			}
		}

		pos, is := 0, 0
		m := uint8(1)
		q := qs
		for range 2 {
			shift := 0
			for range 4 {
				dl := dAll * float32(sc[is])
				is++
				for l := range 16 {
					out[pos] = dl * float32(int8((q[l]>>shift)&3)-qhOffset(hmask[l], m))
					pos++
				}
				dl = dAll * float32(sc[is])
				is++
				for l := range 16 {
					out[pos] = dl * float32(int8((q[16+l]>>shift)&3)-qhOffset(hmask[16+l], m))
					pos++
				}
				shift += 2
				m <<= 1
			}
			q = q[32:]
		}
	}
	return nil
}
