#ifndef _CONFIG_H_
#define _CONFIG_H_

#define PRINT_PERFORMANCE

/* ===
	A partir de là, vous ne devriez pas toucher, sauf si
	vous savez ce que vous faites
   ===
*/

/* Défini les macros BIG_ENDIAN et LITTLE_ENDIAN */
#ifdef __BYTE_ORDER__
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#undef BIG_ENDIAN
#define BIG_ENDIAN
#endif
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
#undef LITTLE_ENDIAN
#define LITTLE_ENDIAN
#endif
#endif /* __BYTE_ORDER__ */

#endif /* _CONFIG_H_ */